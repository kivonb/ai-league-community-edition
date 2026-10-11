"""Local integration: actual API/runner/tool, fake model decisions and AWS clients.

No credentials, network requests, database writes or paid inference are needed.
"""
import ast
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch
import zipfile
from decimal import Decimal

import boto3
import yaml

ROOT = Path(__file__).resolve().parents[3]
API = ROOT / 'lambda' / 'agentic-api'
TOOL = ROOT / 'lambda' / 'pathfinder-tool'
sys.path.insert(0, str(API))
sys.path.append(str(TOOL))

env = {name: 'offline-test' for name in ('GAME_SESSIONS_TABLE', 'LEADERBOARD_TABLE',
       'SUBMISSIONS_TABLE', 'AGENT_CONFIGURATIONS_TABLE', 'MAPS_TABLE')}
with patch.dict(os.environ, env), patch.object(boto3, 'resource'), patch.object(boto3, 'client'):
    import index as api
    import agent_config_handlers as config
    import game_runner
    import game_runner_handler as runner
    import path_parser
    import prompt_formatter

spec = importlib.util.spec_from_file_location('navigation_pathfinder', TOOL / 'index.py')
pathfinder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pathfinder)
SCHEMA = json.loads((TOOL / 'tool-schema.json').read_text())


def validate(arguments, schema):
    """Check the small JSON-schema subset used by this actual gateway contract."""
    types = {'object': dict, 'string': str, 'integer': int, 'number': (int, float)}
    assert isinstance(arguments, types[schema['Type']])
    if schema['Type'] == 'object':
        assert all(key in arguments for key in schema.get('Required', []))
        assert set(arguments) <= set(schema['Properties'])
        for key, value in arguments.items():
            validate(value, schema['Properties'][key])


class NavigationIntegrationTests(unittest.TestCase):
    def fixture(self):
        return {'grid': [['start', 'normal', 'c5', 'treasure'], ['c7', 'wall', 'normal', 'normal']],
                'playerStart': {'row': 0, 'col': 0, 'expectedAnswer': 'HIDDEN_START'},
                'defaults': {'timeLimit': 65, 'lives': 5, 'expectedAnswer': 'HIDDEN_DEFAULT'},
                'tileOverrides': {'c5': {'points': 400, 'damage': 1, 'question': 'HIDDEN_OVERRIDE'}},
                'challenges': {'0,2': {'question': 'What follows Monday?', 'expectedAnswer': 'Tuesday',
                                     'gradingStrategy': 'exact_match'}},
                'challengeAssignments': {'0,2': {'expectedAnswer': 'HIDDEN_ASSIGNMENT'}},
                'expectedAnswer': 'HIDDEN_ROOT'}

    def api_event(self, map_data, prompt):
        table = MagicMock()
        saved = {'SUPERVISOR': {**config.DEFAULT_SUPERVISOR_CONFIG,
                               'subAgents': [config.DEFAULT_PATHFINDER_SUBAGENT_ID]},
                 f'SUBAGENT#{config.DEFAULT_PATHFINDER_SUBAGENT_ID}': config.DEFAULT_PATHFINDER_SUBAGENT}
        table.get_item.side_effect = lambda Key: {'Item': saved.get(Key['sk'], {})}
        client = MagicMock()
        with patch.dict(os.environ, {'AGENT_RUNTIME_ARN': 'mock-runtime', 'GAME_RUNNER_FUNCTION': 'mock-runner'}), \
                patch.object(api, 'agent_configurations_table', table), \
                patch.object(api, 'game_sessions_table', MagicMock()), \
                patch.object(boto3, 'client', return_value=client):
            response = api.handle_invoke_agent_core_runtime(
                {'mapId': 'offline', 'mapData': json.dumps(map_data), 'navigationPath': prompt},
                {'identity': {'sub': 'offline-user'}})
        return response, client

    def run_pipeline(self, map_data, strategy='score_aware', planning=None):
        prompt = f'Use strategy {strategy}.'
        if planning:
            prompt += f' Use planning settings {json.dumps(planning)}.'
        response, client = self.api_event(map_data, prompt)
        self.assertEqual(response['status'], 'starting')
        event = json.loads(client.invoke.call_args.kwargs['Payload'])
        updates, tools = [], []

        def fake_navigation(runtime_arn, payload, **kwargs):
            self.assertEqual(payload['task_type'], 'pathfinding')
            original = payload['prompt']
            for secret in ['HIDDEN_', 'expectedAnswer', 'What follows Monday?', 'Tuesday']:
                self.assertNotIn(secret, original)
            block = original.split('<navigation_map>', 1)[1].split('</navigation_map>', 1)[0]
            arguments = {'game_map': block, 'strategy': strategy}
            if planning:
                arguments['planning'] = planning
            validate(arguments, SCHEMA['InputSchema'])
            result = pathfinder.lambda_handler(arguments, None)
            tools.append(result)
            body = json.loads(result['body'])
            answer = json.dumps(body['path']) if result['statusCode'] == 200 else json.dumps(body)
            return answer, {'total_usage': {'outputTokens': 1000}}

        with patch.object(boto3, 'resource'), \
                patch.object(runner, '_update_session', side_effect=lambda table, sid, value: updates.append(value)), \
                patch.object(runner, 'invoke_agent_runtime', side_effect=fake_navigation) as nav, \
                patch.object(game_runner, 'invoke_agent_runtime',
                             return_value=('Tuesday', {'total_usage': {'outputTokens': 2}})) as answers, \
                patch('time.sleep') as sleep:
            runner.handler(event, None)
        self.assertEqual(nav.call_count, 1)
        sleep.assert_not_called()
        return updates[-1], json.loads(tools[-1]['body']), answers.call_count

    def test_score_aware_api_gateway_and_game_flow(self):
        data = self.fixture()
        original = copy.deepcopy(data)
        final, tool, calls = self.run_pipeline(data)
        self.assertEqual(final['status'], 'completed')
        self.assertTrue(final['reachedTreasure'])
        self.assertEqual(final['qaScore'], 650)
        self.assertEqual(final['livesRemaining'], 5)
        self.assertEqual(calls, 1)
        self.assertEqual(tool['steps'], 5)
        self.assertEqual(tool['planning']['timeLimitSeconds'], 65)
        self.assertEqual(data, original)

    def test_explicit_settings_reach_tool(self):
        final, tool, calls = self.run_pipeline(self.fixture(), planning={'maxExtraMoves': 0})
        self.assertEqual(tool['steps'], 3)
        self.assertEqual(final['qaScore'], 400)
        self.assertEqual(calls, 1)

    def test_old_strategies_still_work(self):
        for strategy in ['swift', 'get_coins']:
            with self.subTest(strategy=strategy):
                final, tool, _ = self.run_pipeline(self.fixture(), strategy=strategy)
                self.assertTrue(final['reachedTreasure'])
                self.assertNotIn('planning', tool)
        legacy = pathfinder.lambda_handler({'game_map': [['start', 'treasure']]}, None)
        self.assertEqual(json.loads(legacy['body'])['path'], ['right'])

    def test_budget_error_propagates_without_retry_or_play(self):
        data = self.fixture()
        data['defaults']['timeLimit'] = 12
        final, tool, calls = self.run_pipeline(data)
        self.assertEqual(final['status'], 'error')
        self.assertIn('finishing route', final['error'])
        self.assertEqual(calls, 0)
        self.assertNotIn('path', tool)

    def test_error_words_never_become_movements(self):
        self.assertIsNone(path_parser.parse_navigation_path(
            json.dumps({'error': 'No time left to move up or down', 'path': ['right']})))

    def test_allowlist_decimal_and_malformed_metadata(self):
        data = self.fixture()
        data['defaults']['timeLimit'] = Decimal('65')
        safe = prompt_formatter.navigation_map(data)
        self.assertEqual(safe['defaults']['timeLimit'], 65)
        self.assertNotIn('HIDDEN_', json.dumps(safe))
        for bad in [{'expectedAnswer': 'SECRET'}, 'SECRET', float('nan'), True]:
            data['defaults']['timeLimit'] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                prompt_formatter.format_navigation_prompt(data)
        data = self.fixture()
        data['grid'][0][0] = {'expectedAnswer': 'SECRET'}
        response, client = self.api_event(data, 'Use score_aware')
        self.assertEqual(response['status'], 'error')
        client.invoke.assert_not_called()

    def test_start_fallback_and_time_alias(self):
        data = {'grid': [['normal', 'start', 'treasure']], 'defaults': {'time': 65}}
        safe = json.loads(prompt_formatter.format_navigation_prompt(data).split('<navigation_map>')[1].split('</navigation_map>')[0])
        self.assertEqual(safe['playerStart'], {'row': 0, 'col': 1})
        result = pathfinder.lambda_handler({'strategy': 'score_aware', 'game_map': safe}, None)
        self.assertEqual(result['statusCode'], 200)
        self.assertEqual(json.loads(result['body'])['path'], ['right'])

    def test_supervisor_forwarding_preserves_metadata(self):
        # Execute the real forwarding function without loading the model SDK or calling a model.
        source = (ROOT / 'lambda' / 'agent-runtime' / 'orchestrator_agent.py').read_text(encoding='utf-8')
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'create_sub_agent_tool')
        calls = []
        namespace = {'tool': lambda function: function, 'logger': MagicMock(),
                     'call_sub_agent': lambda *args: calls.append(args) or '[]'}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<real-forwarder>', 'exec'), namespace)
        original = prompt_formatter.format_navigation_prompt(self.fixture()) + ' Use score_aware.'
        forward = namespace['create_sub_agent_tool']('id', 'description', original_prompt=original)
        forward('Reformatted navigation request')
        self.assertEqual(calls[-1][1], original)

    def test_seed_sources_and_schema_are_consistent(self):
        seed = yaml.safe_load((ROOT / 'agent-config' / 'config.yaml').read_text())
        for prompt in [seed['subAgents'][0]['systemPrompt'], config.DEFAULT_PATHFINDER_SUBAGENT['systemPrompt']]:
            self.assertIn('score_aware', prompt)
            self.assertIn('<navigation_map>', prompt)
            self.assertIn('default', prompt.lower())
        for file in ['index.py', 'score_aware.py', 'tool-schema.json']:
            self.assertEqual((TOOL / file).read_text(),
                             (ROOT / 'agent-config' / 'tools' / 'Pathfinder' / file).read_text())
        content = io.BytesIO()
        with zipfile.ZipFile(content, 'w') as archive:
            archive.writestr('tool-schema.json', json.dumps(SCHEMA))
        with zipfile.ZipFile(content) as archive:
            schemas = config._packaged_pathfinder_schema(archive, 'AgentCoreGatewayTool-Pathfinder')
        self.assertEqual(schemas[0]['inputSchema']['properties']['planning']['type'], 'object')
        self.assertIn('score_aware', schemas[0]['description'])

    def test_schema_auto_update_uses_packaged_contract_without_model(self):
        content = io.BytesIO()
        with zipfile.ZipFile(content, 'w') as archive:
            archive.writestr('tool-schema.json', json.dumps(SCHEMA))
            archive.writestr('index.py', (TOOL / 'index.py').read_text())
        lambda_client, gateway = MagicMock(), MagicMock()
        lambda_client.get_function.return_value = {'Code': {'Location': 'https://mock.invalid/code'}}
        gateway.list_gateway_targets.return_value = {'items': [{'name': 'AgentCoreGatewayTool-Pathfinder', 'targetId': 'target'}]}
        gateway.get_gateway_target.return_value = {'targetConfiguration': {'mcp': {'lambda': {'lambdaArn': 'mock-lambda'}}}}
        def client(service):
            if service == 'lambda':
                return lambda_client
            if service == 'bedrock-agentcore-control':
                return gateway
            raise AssertionError(f'Unexpected service call: {service}')
        with patch.dict(os.environ, {'GATEWAY_URL': 'https://mock.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp'}), \
                patch.object(boto3, 'client', side_effect=client), \
                patch('urllib.request.urlopen') as download:
            download.return_value.__enter__.return_value.read.return_value = content.getvalue()
            config._auto_update_gateway_schema('AgentCoreGatewayTool-Pathfinder')
        gateway.update_gateway_target.assert_called_once()
        schemas = gateway.update_gateway_target.call_args.kwargs['targetConfiguration']['mcp']['lambda']['toolSchema']['inlinePayload']
        self.assertIn('score_aware', schemas[0]['description'])
        self.assertEqual(schemas[0]['inputSchema']['properties']['planning']['properties']['maxExtraMoves']['type'], 'integer')

    def test_london_finale_route_through_actual_flow(self):
        grid = [['normal']*10 for _ in range(10)]
        grid[0] = ['c17', 'c7', 'c7', 'c7', 'c17', 'normal', 'normal', 'normal', 'c2', 'treasure']
        grid[4] = ['c17'] + ['c8']*8 + ['c17']
        for r in range(5, 9):
            grid[r][9] = 'c7'
        grid[7][:3] = ['c5']*3
        grid[8][2] = grid[9][2] = 'c5'
        grid[9][4] = grid[9][9] = 'c17'
        data = {'grid': grid, 'playerStart': {'row': 9, 'col': 0},
                'defaults': {'timeLimit': 65}, 'tileOverrides': {'c17': {'points': 50}},
                'challenges': {f'{r},{c}': {'question': 'What follows Monday?', 'expectedAnswer': 'Tuesday',
                                          'gradingStrategy': 'exact_match'}
                               for r, row in enumerate(grid) for c, tile in enumerate(row)
                               if tile in ('c5', 'c17', 'c2')}}
        final, tool, calls = self.run_pipeline(data)
        self.assertEqual(final['qaScore'], 2500)
        self.assertTrue(final['reachedTreasure'])
        self.assertEqual(final['livesRemaining'], 5)
        self.assertEqual(calls, 5)
        self.assertEqual(tool['steps'], 38)


if __name__ == '__main__':
    unittest.main()

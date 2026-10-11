"""Regression coverage for original-map serialization in actual v2 game flow."""
import json
import unittest
from unittest.mock import patch
from game_runner import run_game_session_v2


class ChallengeContextTests(unittest.TestCase):
    def test_normal_and_key_door_challenges_reach_treasure(self):
        for tiles in [['normal', 'c5', 'treasure'], ['normal', 'c40', 'c30', 'c5', 'treasure']]:
            with self.subTest(tiles=tiles):
                grid = [tiles]
                challenges = {f'0,{i}': {'question': 'What day comes after Monday?',
                              'expectedAnswer': 'Tuesday', 'gradingStrategy': 'exact_match'}
                              for i in range(1, len(tiles)-1)}
                calls = []

                def invoke(runtime_arn, payload, **kwargs):
                    calls.append(payload)
                    self.assertEqual(json.loads(payload['context'])['grid'], grid)
                    self.assertEqual(payload['task_type'], 'challenge')
                    return 'Tuesday', {'total_usage': {'outputTokens': 2}}

                with patch('game_runner.invoke_agent_runtime', side_effect=invoke):
                    result = run_game_session_v2(
                        session_id='regression', map_data={'grid': grid, 'challenges': challenges},
                        navigation_path=[(0, i) for i in range(1, len(tiles))],
                        custom_model_count=0, runtime_arn='test-runtime', db_flush_fn=lambda *args: None)
                self.assertTrue(result['reachedTreasure'])
                self.assertEqual(len(calls), len(tiles)-2)
                self.assertEqual(sum(event['type'] == 'WinChallenge' for event in result['gameEvents']), len(tiles)-2)


if __name__ == '__main__':
    unittest.main()

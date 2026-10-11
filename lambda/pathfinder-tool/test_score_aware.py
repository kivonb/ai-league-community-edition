import copy
import json
import random
import unittest

import index
from score_aware import plan_score_aware, MOVES, BLOCKED


class ScoreAwareTests(unittest.TestCase):
    def plan(self, grid, **options):
        data = {'grid': grid, 'defaults': {'timeLimit': 65}}
        start = next((r, c) for r, row in enumerate(grid) for c, t in enumerate(row) if t == 'start')
        goal = next((r, c) for r, row in enumerate(grid) for c, t in enumerate(row) if t == 'treasure')
        result = plan_score_aware(data, start, goal, options)
        position = start
        for i, move in enumerate(result['path']):
            dr, dc = next((dr, dc) for dr, dc, name in MOVES if name == move)
            position = (position[0]+dr, position[1]+dc)
            self.assertNotIn(grid[position[0]][position[1]], BLOCKED)
            if position == goal:
                self.assertEqual(i, len(result['path'])-1)
        self.assertEqual(position, goal)
        self.assertTrue(result['feasible'])
        self.assertLessEqual(result['estimatedSeconds'], 65-result['reserveSeconds'])
        self.assertLessEqual(result['moves'], result['maxMoves'])
        return result

    def test_profitable_detour_and_count_once(self):
        result = self.plan([['start', 'normal', 'treasure'], ['c7', 'wall', 'wall']])
        self.assertEqual(result['moves'], 4)
        self.assertEqual(result['tileCounts'], {'c7': 1})
        self.assertEqual(result['expectedTilePoints'], 250)
        self.assertEqual(result['estimatedSeconds'], 10.6)
        self.assertGreater(result['expectedScore'], result['baseline']['expectedScore'])

    def test_move_cap_disallows_detour(self):
        result = self.plan([['start', 'normal', 'treasure'], ['c7', 'wall', 'wall']], maxExtraMoves=0)
        self.assertEqual(result['moves'], 2)
        self.assertEqual(result['expectedTilePoints'], 0)

    def test_no_reward_after_treasure(self):
        result = self.plan([['start', 'treasure', 'c7']])
        self.assertEqual(result['path'], ['right'])
        self.assertNotIn('c7', result['tileCounts'])

    def test_spikes_and_doors_are_blocked(self):
        result = self.plan([['start', 'c8', 'treasure'], ['normal', 'normal', 'normal']])
        self.assertEqual(result['moves'], 4)
        with self.assertRaisesRegex(ValueError, 'No spike-free'):
            self.plan([['start', 'c30', 'treasure']])

    def test_slow_detour_rejected(self):
        result = self.plan([['start', 'normal', 'treasure'], ['c5', 'wall', 'wall']],
                           challengeProfiles={'c5': {'seconds': 60}})
        self.assertEqual(result['moves'], 2)

    def test_low_budget_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'finishing route'):
            self.plan([['start', 'treasure']], timeLimitSeconds=10, reserveSeconds=9,
                      overheadSeconds=2)

    def test_override_and_hidden_answers_ignored(self):
        data = {'grid': [['start', 'c5', 'treasure']], 'defaults': {'timeLimit': 65},
                'tileOverrides': {'c5': {'points': 400, 'damage': 2}},
                'challengeAssignments': {'0,1': {'expectedAnswer': 'secret'}}}
        result = plan_score_aware(data, (0, 0), (0, 2))
        self.assertEqual(result['expectedTilePoints'], 392)
        self.assertEqual(result['expectedLifeLoss'], .04)
        modified = copy.deepcopy(data)
        modified['challengeAssignments'] = {'0,1': {'expectedAnswer': 'different'}}
        self.assertEqual(result, plan_score_aware(modified, (0, 0), (0, 2)))

    def test_handler_and_metadata_requirement(self):
        data = {'grid': [['start', 'c7', 'treasure']], 'defaults': {'timeLimit': 65}}
        response = index.lambda_handler({'strategy': 'score_aware', 'game_map': data}, None)
        self.assertEqual(response['statusCode'], 200)
        body = json.loads(response['body'])
        self.assertEqual(body['path'], ['right', 'right'])
        self.assertEqual(body['planning']['tileCounts']['c7'], 1)
        response = index.lambda_handler({'strategy': 'score_aware', 'game_map': data['grid']}, None)
        self.assertEqual(response['statusCode'], 400)
        data['defaults']['timeLimit'] = 0
        self.assertEqual(index.lambda_handler({'strategy': 'score_aware', 'game_map': data}, None)['statusCode'], 400)

    def test_invalid_estimates_and_search_bound(self):
        for value in [True, float('nan'), -1]:
            with self.assertRaises(ValueError):
                self.plan([['start', 'treasure']], moveSeconds=value)
        result = self.plan([['start', 'normal', 'treasure'], ['c7', 'c7', 'c7']],
                           maxCandidateEvaluations=1)
        self.assertEqual(result['candidateEvaluations'], 1)
        self.assertTrue(result['searchLimitReached'])

    def test_generated_maps_keep_route_invariants(self):
        rng = random.Random(42)
        for _ in range(100):
            grid = [[rng.choice(['normal', 'c7', 'c5', 'wall', 'c8', 'c30'])
                     for _ in range(4)] for _ in range(4)]
            # Guarantee one safe finishing corridor; other cells remain random.
            for i in range(4):
                grid[0][i] = grid[i][3] = 'normal'
            grid[0][0], grid[3][3] = 'start', 'treasure'
            with self.subTest(grid=grid):
                self.plan(grid)


if __name__ == '__main__':
    unittest.main()

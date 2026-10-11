import itertools
import json
import random
import unittest
import index as pathfinder


class SwiftRouteTests(unittest.TestCase):
    def test_tie_prefers_reward(self):
        grid = [['normal', 'normal'], ['c7', 'treasure']]
        self.assertEqual(pathfinder.swift_path(grid, 2, 2, (0, 0), (1, 1)), ['down', 'right'])

    def test_tie_prefers_safety_over_points(self):
        grid = [['normal', 'c8', 'c7'], ['normal', 'wall', 'treasure'], ['normal', 'normal', 'normal']]
        # The safe route is longer, so swift deliberately does NOT take it.
        self.assertEqual(len(pathfinder.swift_path(grid, 3, 3, (0, 0), (1, 2))), 3)
        grid = [['normal', 'c8'], ['normal', 'treasure']]
        self.assertEqual(pathfinder.swift_path(grid, 2, 2, (0, 0), (1, 1)), ['down', 'right'])

    def test_map_specific_points(self):
        grid = [['normal', 'c17'], ['c5', 'treasure']]
        self.assertEqual(pathfinder.swift_path(grid, 2, 2, (0, 0), (1, 1)), ['right', 'down'])
        self.assertEqual(pathfinder.swift_path(grid, 2, 2, (0, 0), (1, 1), {'c17': {'points': 50}}), ['down', 'right'])

    def test_handler_passes_map_specific_points(self):
        response = pathfinder.lambda_handler({
            'game_map': {'grid': [['normal', 'c17'], ['c5', 'treasure']],
                         'tileOverrides': {'c17': {'points': 50}},
                         'playerStart': {'row': 0, 'col': 0}},
            'strategy': 'swift'}, None)
        self.assertEqual(response['statusCode'], 200)
        self.assertEqual(json.loads(response['body'])['path'], ['down', 'right'])

    def test_unreachable_and_start_at_goal(self):
        self.assertEqual(pathfinder.swift_path([['normal', 'wall', 'treasure']], 1, 3, (0, 0), (0, 2)), [])
        self.assertEqual(pathfinder.swift_path([['treasure']], 1, 1, (0, 0), (0, 0)), [])

    def test_against_exhaustive_shortest_routes(self):
        rng = random.Random(42)
        for _ in range(80):
            grid = [[rng.choice(['normal', 'c7', 'c5', 'c8', 'c17']) for _ in range(3)] for _ in range(3)]
            grid[0][0], grid[2][2] = 'normal', 'treasure'
            candidates = set(itertools.permutations(['down', 'down', 'right', 'right']))

            def rank(moves):
                r = c = 0
                damage = points = distractions = 0
                for move in moves:
                    r += move == 'down'
                    c += move == 'right'
                    tile = grid[r][c]
                    damage += tile == 'c8'
                    points += pathfinder.SWIFT_REWARD_POINTS.get(tile, 0)
                    distractions += tile == 'c17'
                return damage, -points, distractions

            result = pathfinder.swift_path(grid, 3, 3, (0, 0), (2, 2))
            self.assertEqual(len(result), 4)
            self.assertEqual(rank(result), min(rank(candidate) for candidate in candidates))


if __name__ == '__main__':
    unittest.main()

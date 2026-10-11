"""Bounded, deterministic detour planning. Never reads question/answer contents."""
import heapq
import math

MOVES = [(-1, 0, 'up'), (1, 0, 'down'), (0, -1, 'left'), (0, 1, 'right')]
BLOCKED = {'wall', 'c8', 'c30', 'c31', 'c32', 'c33'}
POINTS = {'c1': 400, 'c2': 600, 'c3': 550, 'c4': 800, 'c5': 250,
          'c6': 1000, 'c7': 250, 'c17': 750, 'c18': 500,
          'c40': 50, 'c41': 50, 'c42': 50, 'c43': 50}
DAMAGE = {'c1': 2, 'c2': 1, 'c3': 1, 'c4': 1, 'c5': 1, 'c6': 2,
          'c17': 2, 'c18': 1}
# Starting estimates, NOT measurements. Caller can supply actual per-type profiles.
PROFILES = {
    'c1': (2, 0, .95), 'c2': (5, 60, .90), 'c3': (4, 35, .90),
    'c4': (7, 100, .85), 'c5': (3, 4, .98), 'c6': (10, 250, .75),
    'c17': (5, 220, .95), 'c18': (6, 120, .85),
    'c40': (3, 10, .90), 'c41': (3, 10, .90),
    'c42': (3, 10, .90), 'c43': (3, 10, .90),
}


def _number(value, name, minimum=0, maximum=100000):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite number')
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be between {minimum} and {maximum}')
    return value


def plan_score_aware(map_data, start, treasure, options=None):
    options = options or {}
    grid = map_data['grid']
    if (not grid or len(grid) > 30 or not grid[0] or len(grid[0]) > 30
            or any(len(row) != len(grid[0]) for row in grid)):
        raise ValueError('score_aware requires a rectangular grid of at most 30 by 30')
    rows, cols = len(grid), len(grid[0])
    defaults = map_data.get('defaults', {})
    overrides = map_data.get('tileOverrides', {})
    budget = _number(options.get('timeLimitSeconds', map_data.get('timeLimit') or
                     defaults.get('timeLimit') or defaults.get('time', 0)), 'timeLimitSeconds', 1, 3600)
    reserve = _number(options.get('reserveSeconds', max(10, budget*.2)), 'reserveSeconds', 0, budget)
    overhead = _number(options.get('overheadSeconds', 10), 'overheadSeconds', 0, budget)
    move_seconds = _number(options.get('moveSeconds', .15), 'moveSeconds', .001, 10)
    extra_moves = int(_number(options.get('maxExtraMoves', 20), 'maxExtraMoves', 0, 60))
    iterations = int(_number(options.get('maxDetours', 12), 'maxDetours', 0, 20))
    evaluation_limit = int(_number(options.get('maxCandidateEvaluations', 5000),
                                   'maxCandidateEvaluations', 1, 20000))
    nav_tokens = _number(options.get('navigationOutputTokens', 1000), 'navigationOutputTokens')
    lives = _number(map_data.get('startingLives', defaults.get('lives', 5)), 'startingLives', 1, 100)
    profiles = {}
    for tile, (seconds, tokens, probability) in PROFILES.items():
        custom = options.get('challengeProfiles', {}).get(tile, {})
        profiles[tile] = (
            _number(custom.get('seconds', seconds), f'{tile}.seconds', 0, 300),
            _number(custom.get('outputTokens', tokens), f'{tile}.outputTokens'),
            _number(custom.get('successProbability', probability), f'{tile}.successProbability', 0, 1))

    def usable(position, destination):
        r, c = position
        return (0 <= r < rows and 0 <= c < cols and grid[r][c] not in BLOCKED
                and (position != treasure or destination == treasure))

    if not usable(start, treasure) or not usable(treasure, treasure):
        raise ValueError('Start or treasure is blocked or outside the grid')
    if sum(cell == 'treasure' for row in grid for cell in row) != 1:
        raise ValueError('Exactly one treasure is required')

    def shortest(source, destination, collected=None):
        # Dijkstra chooses quick connectors; treasure is impassable unless it is the goal.
        weighted = collected is not None
        frontier = [(0, source)]
        costs, parent = {source: 0}, {}
        while frontier:
            cost, position = heapq.heappop(frontier)
            if cost != costs[position]:
                continue
            if position == destination:
                route = [destination]
                while route[-1] != source:
                    route.append(parent[route[-1]])
                return list(reversed(route))
            r, c = position
            for dr, dc, _ in MOVES:
                neighbor = (r+dr, c+dc)
                if not usable(neighbor, destination):
                    continue
                nr, nc = neighbor
                time = profiles.get(grid[nr][nc], (0, 0, 1))[0]
                step = (move_seconds + (time if neighbor not in collected else 0)) if weighted else 1
                candidate = cost + step
                if candidate < costs.get(neighbor, float('inf')):
                    costs[neighbor], parent[neighbor] = candidate, position
                    heapq.heappush(frontier, (candidate, neighbor))
        return None

    baseline = shortest(start, treasure)
    if baseline is None:
        raise ValueError('No spike-free, door-free route to treasure; use a separately reviewed strategy')
    maximum_moves = len(baseline)-1 + extra_moves
    available = budget-reserve

    def assess(route):
        seen = {start}
        points = loss = count = 0
        tokens = nav_tokens
        seconds = overhead + (len(route)-1)*move_seconds
        tile_counts = {}
        for position in route[1:]:
            if position in seen:
                continue
            seen.add(position)
            r, c = position
            tile = grid[r][c]
            if tile not in POINTS:
                continue
            tile_counts[tile] = tile_counts.get(tile, 0)+1
            raw_points = _number(overrides.get(tile, {}).get('points', POINTS[tile]), f'{tile}.points')
            time, output, success = profiles.get(tile, (0, 0, 1))
            points += raw_points*success
            seconds += time
            tokens += output
            if tile in profiles:
                count += 1
                damage = _number(overrides.get(tile, {}).get('damage', DAMAGE.get(tile, 0)), f'{tile}.damage')
                loss += damage*(1-success)
        multiplier = _number(defaults.get('livesBonusMultiplier', 250), 'livesBonusMultiplier')
        token_base = _number(defaults.get('tokenBonus', 1000), 'tokenBonus')
        bonus = max(0, token_base-tokens/count) if count else token_base
        estimated_score = (points + max(0, lives-loss)*multiplier + bonus
                           + _number(defaults.get('treasureBonus', 1000), 'treasureBonus'))
        return {'expectedScore': round(estimated_score, 3), 'expectedTilePoints': round(points, 3),
                'estimatedSeconds': round(seconds, 3), 'estimatedOutputTokens': tokens,
                'expectedLifeLoss': round(loss, 3), 'challengeCount': count,
                'tileCounts': tile_counts, 'moves': len(route)-1,
                'feasible': seconds <= available and len(route)-1 <= maximum_moves and loss < lives}

    # Compare shortest and quick-connector seeds. Neither is allowed to exceed the budget.
    seeds = [baseline, shortest(start, treasure, {start})]
    feasible = [(route, assess(route)) for route in seeds if route is not None and assess(route)['feasible']]
    if not feasible:
        raise ValueError('Even the finishing route exceeds the estimated budget after the safety reserve')
    route, current = max(feasible, key=lambda pair: (pair[1]['expectedScore'], -pair[1]['estimatedSeconds']))
    targets = [(r, c) for r, row in enumerate(grid) for c, tile in enumerate(row)
               if tile in POINTS and tile not in BLOCKED]
    if len(targets) > 80:
        raise ValueError('At most 80 reward tiles are supported by this bounded planner')
    accepted = []
    evaluations = 0
    for _ in range(iterations):
        visited = set(route)
        winner = None
        connectors = {}

        def connect(source, destination):
            key = (source, destination)
            if key not in connectors:
                connectors[key] = shortest(source, destination, visited)
            return connectors[key]

        for target in targets:
            if target in visited:
                continue
            for slot in range(len(route)-1):
                if evaluations >= evaluation_limit:
                    break
                evaluations += 1
                first, second = connect(route[slot], target), connect(target, route[slot+1])
                if first is None or second is None:
                    continue
                candidate = route[:slot] + first + second[1:] + route[slot+2:]
                metrics = assess(candidate)
                if not metrics['feasible'] or metrics['expectedScore'] <= current['expectedScore']:
                    continue
                rank = (metrics['expectedScore'], -metrics['estimatedSeconds'], -metrics['moves'])
                if winner is None or rank > winner[0]:
                    winner = (rank, candidate, metrics, target)
        if winner is None:
            break
        _, route, current, target = winner
        accepted.append({'target': list(target), 'moves': current['moves'],
                         'expectedScore': current['expectedScore'], 'estimatedSeconds': current['estimatedSeconds']})
        if evaluations >= evaluation_limit:
            break

    path = []
    for first, second in zip(route, route[1:]):
        path.append(next(move for dr, dc, move in MOVES
                         if (first[0]+dr, first[1]+dc) == second))
    return {'path': path, **current, 'baseline': assess(baseline), 'acceptedDetours': accepted,
            'timeLimitSeconds': budget, 'reserveSeconds': reserve, 'maxMoves': maximum_moves,
            'candidateEvaluations': evaluations, 'searchLimitReached': evaluations >= evaluation_limit,
            'assumptions': {'moveSeconds': move_seconds, 'overheadSeconds': overhead,
                            'navigationOutputTokens': nav_tokens,
                            'challengeProfiles': {key: {'seconds': value[0], 'outputTokens': value[1],
                                                       'successProbability': value[2]} for key, value in profiles.items()}},
            'limitations': 'Estimates, not guarantees; bounded greedy search, not global optimum. Spikes and all doors are blocked. No online replanning or hidden answers used.'}

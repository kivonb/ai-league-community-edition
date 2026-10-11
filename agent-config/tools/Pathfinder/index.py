# Lambda

import json
from score_aware import plan_score_aware
from collections import deque

CELL_POINTS = {"c7": 250}
COLLECTIBLE_COINS = {"c7"}
DIRECTIONS = [(-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")]

# Potential rewards, not guaranteed winnings: challenges still require correct answers.
# These match the game runner defaults; full map objects can override them.
SWIFT_REWARD_POINTS = {
    "c1": 400, "c2": 600, "c3": 550, "c4": 800, "c5": 250,
    "c6": 1000, "c7": 250, "c17": 750, "c18": 500,
    "c30": 1000, "c31": 1000, "c32": 1000, "c33": 1000,
    "c40": 50, "c41": 50, "c42": 50, "c43": 50,
}


def lambda_handler(event, context):
    """
    AWS Lambda function for pathfinding using Swift path strategy by default
    Handles both API Gateway format and direct AgentCore Gateway format

    ## Map Definitions
        - "wall": non-walkable cell
        - "treasure": target cell to reach
        - "normal": walkable cell with no special properties
        - "start": the start cell of your avatar, acts as normal cell
        - "c1": Violent Violet guardrail challenge
        - "c2": Blue Brain code execution challenge
        - "c3": Memento memory challenge
        - "c4": Dark Prophet web scraping challenge
        - "c5": Bonehead challenge, simple question that requires little skill
        - "c6": Boss Challenge, most requires all skills
        - "c7": Coins that increase score when collected with no challenge
        - "c8": Spikes that reduce health traveled over

    ## Map JSON Example
    [
        ["start","normal","c5","normal","normal","normal","c5","normal","normal","c1"],
        ["normal","wall","wall","normal","wall","wall","wall","wall","wall","normal"],
        ["c8","wall","wall","c5","wall","c7","c7","c7","wall","c3"],
        ["normal","wall","c8","normal","wall","c8","wall","c8","wall","normal"],
        ["normal","wall","c7","normal","wall","normal","normal","normal","wall","normal"],
        ["c5","wall","c7","normal","wall","c5","wall","normal","wall","c5"],
        ["normal","wall","c7","normal","wall","normal","wall","normal","wall","normal"],
        ["c1","wall","c8","normal","c2","normal","wall","normal","c4","normal"],
        ["normal","wall","wall","wall","wall","wall","wall","normal","normal","c7"],
        ["c7","normal","c3","normal","c4","normal","c2","normal","treasure","normal"]
    ]

    ## Pathfinding Lambda with strategy selection.

    Usage: Use strategy get_coins

    Strategies:
      swift     - Shortest path; ties favor fewer spikes, then higher potential rewards
      get_coins - Greedily collect c7 coins on the way to treasure
      score_aware - Bounded reward detours using full map metadata and time estimates
    """

    try:
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event

        print(f"DEBUG: start_pos={body.get('start_pos')} grid_size={len(str(body.get('game_map','')))} chars")
        game_map = body.get('game_map', [])
        start_pos = body.get('start_pos', [0, 0])
        strategy = body.get('strategy', 'swift')
        if strategy not in ('swift', 'get_coins', 'score_aware'):
            return _err(400, 'Unsupported strategy; choose swift, get_coins or score_aware')
        print(f"DEBUG: strategy={strategy}")

        # Robustly parse game_map — handle various formats from LLM sub-agents:
        # 1. JSON string of 2D array: '[[...],[...]]'
        # 2. JSON string of object with "grid" key: '{"grid": [[...],...]}'
        # 3. JSON string of object with "game_map" key: '{"game_map": [[...],...]}'
        # 4. Already a list (2D array)
        # 5. Already a dict with "grid" key
        # 6. Malformed string with leading/trailing junk from LLM extraction errors
        map_object = None  # Keep reference to full map object for playerStart extraction
        if isinstance(game_map, str):
            # Strip leading/trailing whitespace
            game_map = game_map.strip()
            # LLMs sometimes add leading commas or trailing quotes when extracting
            # from larger JSON — find the first [ or { and last ] or }
            first_bracket = -1
            for i, ch in enumerate(game_map):
                if ch in ('[', '{'):
                    first_bracket = i
                    break
            last_bracket = -1
            for i in range(len(game_map) - 1, -1, -1):
                if game_map[i] in (']', '}'):
                    last_bracket = i
                    break
            if first_bracket >= 0 and last_bracket > first_bracket:
                game_map = game_map[first_bracket:last_bracket + 1]
            try:
                game_map = json.loads(game_map)
            except json.JSONDecodeError:
                return _err(400, f'game_map is not valid JSON: {game_map[:100]}')

        if isinstance(game_map, dict):
            map_object = game_map
            # Extract grid from various wrapper formats
            if 'grid' in game_map:
                game_map = game_map['grid']
            elif 'game_map' in game_map:
                game_map = game_map['game_map']
            elif 'map' in game_map:
                game_map = game_map['map']

        # Parse start_pos if it's a string
        if isinstance(start_pos, str):
            try:
                start_pos = json.loads(start_pos)
            except json.JSONDecodeError:
                start_pos = [0, 0]

        # If map_object had a playerStart field, use it (most reliable source)
        if map_object and 'playerStart' in map_object:
            ps = map_object['playerStart']
            if isinstance(ps, dict) and 'row' in ps and 'col' in ps:
                start_pos = [int(ps['row']), int(ps['col'])]
                print(f"DEBUG: Using playerStart from map object: {start_pos}")

        # Fallback: find "start" cell in the grid to determine start position
        if not map_object or 'playerStart' not in (map_object or {}):
            for r_idx, row in enumerate(game_map):
                for c_idx, cell in enumerate(row):
                    if cell == 'start':
                        start_pos = [r_idx, c_idx]
                        print(f"DEBUG: Found 'start' cell at: {start_pos}")
                        break
                else:
                    continue
                break

        if not game_map or not isinstance(game_map, list):
            return _err(400, 'Missing or invalid game_map')

        rows, cols = len(game_map), len(game_map[0])
        treasure = None
        for r in range(rows):
            for c in range(cols):
                if game_map[r][c] == 'treasure':
                    treasure = (r, c)
                    break
            if treasure:
                break

        if not treasure:
            return _err(400, 'No treasure found on map')

        planning_result = None
        if strategy == 'score_aware':
            if map_object is None:
                return _err(400, 'score_aware requires a full map object with time limit and tile overrides, not just a grid')
            try:
                planning_result = plan_score_aware(map_object, tuple(start_pos), treasure,
                                                  body.get('planning', {}))
            except (ValueError, TypeError, KeyError) as error:
                return _err(400, str(error))
            path = planning_result['path']
        elif strategy == 'get_coins':
            path = get_coins_path(game_map, rows, cols, tuple(start_pos), treasure)
        else:
            path = swift_path(game_map, rows, cols, tuple(start_pos), treasure,
                              tile_overrides=(map_object or {}).get('tileOverrides'))

        result = {'path': path, 'steps': len(path), 'start_position': start_pos}
        if planning_result is not None:
            result['planning'] = {key: value for key, value in planning_result.items() if key != 'path'}
        print(f"RESULT: strategy={strategy} steps={len(path)}")
        return {'statusCode': 200, 'body': json.dumps(result)}

    except Exception as e:
        print(f"ERROR: {e}")
        return _err(500, str(e))


def _err(code, msg):
    return {'statusCode': code, 'body': json.dumps({'error': msg})}


def _bfs(game_map, rows, cols, start, goal):
    """BFS shortest path between two points."""
    queue = deque([(start[0], start[1], [])])
    visited = {(start[0], start[1])}
    while queue:
        r, c, path = queue.popleft()
        if (r, c) == goal:
            return path
        for dr, dc, move in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and game_map[nr][nc] != 'wall' and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc, path + [move]))
    return None


def swift_path(game_map, rows, cols, start, treasure, tile_overrides=None):
    """Choose the safest, highest-potential-reward route among shortest paths.

    This is only a tie-breaker, not a time-budgeted detour planner. Challenge
    success, keys/locked doors, and model response times are not predicted here.
    """
    tile_overrides = tile_overrides or {}

    # Reverse BFS measures the minimum number of remaining moves at every cell.
    # Any chosen move MUST reduce that distance by one: no loops or extra steps.
    distances = {treasure: 0}
    queue = deque([treasure])
    ordered = []
    while queue:
        r, c = queue.popleft()
        ordered.append((r, c))
        for dr, dc, _ in DIRECTIONS:
            neighbor = (r + dr, c + dc)
            nr, nc = neighbor
            if (0 <= nr < rows and 0 <= nc < cols
                    and game_map[nr][nc] != "wall" and neighbor not in distances):
                distances[neighbor] = distances[(r, c)] + 1
                queue.append(neighbor)
    if start not in distances:
        return []

    # Distances increase in BFS order, so each shorter suffix is already solved.
    # Rank: least spike damage, most potential points, fewest counting distractions.
    best = {treasure: (0, 0, 0)}
    next_step = {}
    for position in ordered:
        if position == treasure:
            continue
        r, c = position
        winning_rank = None
        for dr, dc, move in DIRECTIONS:
            neighbor = (r + dr, c + dc)
            if distances.get(neighbor) != distances[position] - 1:
                continue
            nr, nc = neighbor
            tile = game_map[nr][nc]
            override = tile_overrides.get(tile, {})
            points = override.get("points", SWIFT_REWARD_POINTS.get(tile, 0))
            damage = override.get("damage", 1) if tile == "c8" else 0
            suffix = best[neighbor]
            rank = (suffix[0] + damage, suffix[1] - points,
                    suffix[2] + (tile == "c17"))
            if winning_rank is None or rank < winning_rank:
                winning_rank = rank
                next_step[position] = (neighbor, move)
        best[position] = winning_rank

    path = []
    position = start
    while position != treasure:
        position, move = next_step[position]
        path.append(move)
    return path


def get_coins_path(game_map, rows, cols, start, treasure):
    """Greedily BFS to best coins-per-step c7 cell, then BFS to treasure."""
    board = [row[:] for row in game_map]
    r, c = start
    full_path = []

    for _ in range(50):
        # BFS to find reachable coins
        queue = deque([(r, c, [])])
        visited = {(r, c)}
        targets = []
        while queue:
            cr, cc, p = queue.popleft()
            if board[cr][cc] in COLLECTIBLE_COINS and (cr, cc) != (r, c):
                dist = max(len(p), 1)
                targets.append((dist, p, cr, cc))  # closest coin first
            for dr, dc, move in DIRECTIONS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != 'wall' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, p + [move]))

        if not targets:
            break
        targets.sort()
        _, path_to, r, c = targets[0]
        full_path.extend(path_to)
        board[r][c] = 'normal'

    # BFS to treasure from current position
    path_end = _bfs(board, rows, cols, (r, c), treasure)
    if path_end is not None:
        full_path.extend(path_end)
        return full_path
    return swift_path(game_map, rows, cols, start, treasure)

import heapq
import math
import time

import numpy as np
import torch


def informed_samples(start, goals, weights, lower, upper, count, bound, generator):
    device = start.device
    dimension = start.numel()
    uniform = lower + torch.rand((count, dimension), device=device, generator=generator) * (upper - lower)
    if not math.isfinite(bound):
        return uniform
    selected = goals[torch.randint(len(goals), (count,), device=device, generator=generator)]
    displacement = (selected - start) * weights
    distance = torch.linalg.vector_norm(displacement, dim=1)
    direction = displacement / distance[:, None].clamp_min(1e-8)
    reflection = direction.clone()
    reflection[:, 0] -= 1.0
    reflection /= torch.linalg.vector_norm(reflection, dim=1)[:, None].clamp_min(1e-8)
    ball = torch.randn((count, dimension), device=device, generator=generator)
    ball /= torch.linalg.vector_norm(ball, dim=1)[:, None].clamp_min(1e-8)
    ball *= torch.rand((count, 1), device=device, generator=generator).pow(1.0 / dimension)
    radii = torch.sqrt((bound * bound - distance.square()).clamp_min(0)) / 2
    ball *= radii[:, None]
    ball[:, 0] *= (bound / 2) / radii.clamp_min(1e-8)
    rotated = ball - 2 * reflection * (ball * reflection).sum(dim=1, keepdim=True)
    samples = (start + selected) / 2 + rotated / weights
    inside = ((samples >= lower) & (samples <= upper)).all(dim=1) & (distance < bound)
    use_informed = torch.rand((count,), device=device, generator=generator) < 0.7
    return torch.where((inside & use_informed)[:, None], samples, uniform)


def batched_bitstar_multi_goal(start, goals, lower, upper, validity, budget_s, seed,
                               batch_size=512, edge_batch_size=64, max_nodes=12000):
    generator = torch.Generator(device=start.device)
    generator.manual_seed(seed)
    weights = validity.weights
    dimension = start.numel()
    nodes = torch.cat((start.reshape(1, -1), goals), dim=0)
    parents = [-1] * len(nodes)
    costs = [0.0] + [math.inf] * len(goals)
    children = [set() for _ in costs]
    goal_indices = set(range(1, len(goals) + 1))
    heuristic = torch.cdist(nodes * weights, goals * weights).min(dim=1).values.cpu().tolist()
    lower_cost = torch.linalg.vector_norm((nodes - start) * weights, dim=1).cpu().tolist()
    vertex_queue = []
    edge_queue = []
    edge_cache = {}
    best_cost = math.inf
    solutions = []
    started = time.perf_counter()
    first_solution_ms = None
    batches = edges_checked = edges_valid = rewires = pruned = 0
    sampling_ms = neighbor_ms = edge_ms = 0.0

    def record_solution(destination):
        nonlocal best_cost, first_solution_ms
        if destination not in goal_indices or costs[destination] >= best_cost:
            return
        best_cost = costs[destination]
        indices = []
        current = destination
        while current >= 0:
            indices.append(current)
            current = parents[current]
        path = nodes[indices[::-1]].detach().cpu().numpy()
        solutions.append((best_cost, destination - 1, path))
        if first_solution_ms is None:
            first_solution_ms = (time.perf_counter() - started) * 1000

    while time.perf_counter() - started < budget_s:
        if best_cost <= heuristic[0] + 1e-7:
            break
        if not vertex_queue and not edge_queue:
            if len(nodes) >= max_nodes:
                break
            sample_started = time.perf_counter()
            samples = informed_samples(start, goals, weights, lower, upper,
                                       min(batch_size, max_nodes - len(nodes)), best_cost, generator)
            estimates = torch.linalg.vector_norm((samples - start) * weights, dim=1)
            estimates += torch.cdist(samples * weights, goals * weights).min(dim=1).values
            samples = samples[estimates < best_cost]
            if len(samples):
                samples = samples[validity.mask(samples)]
            nodes = torch.cat((nodes, samples), dim=0)
            costs.extend([math.inf] * len(samples))
            parents.extend([-1] * len(samples))
            children.extend(set() for _ in range(len(samples)))
            heuristic.extend(torch.cdist(samples * weights, goals * weights).min(dim=1).values.cpu().tolist())
            lower_cost.extend(torch.linalg.vector_norm((samples - start) * weights, dim=1).cpu().tolist())
            vertex_queue = [(cost + heuristic[index], index, cost)
                            for index, cost in enumerate(costs)
                            if math.isfinite(cost) and cost + heuristic[index] < best_cost]
            heapq.heapify(vertex_queue)
            sampling_ms += (time.perf_counter() - sample_started) * 1000
            batches += 1
        if not vertex_queue and not edge_queue:
            continue
        while vertex_queue and (not edge_queue or vertex_queue[0][0] <= edge_queue[0][0]):
            if time.perf_counter() - started >= budget_s:
                break
            priority, source, queued_cost = heapq.heappop(vertex_queue)
            if queued_cost != costs[source] or priority >= best_cost:
                continue
            neighbor_started = time.perf_counter()
            distances = torch.linalg.vector_norm((nodes - nodes[source]) * weights, dim=1)
            count = min(len(nodes), max(32, math.ceil(2 * math.e * (1 + 1 / dimension) * math.log(len(nodes)))))
            neighbors = set(torch.topk(distances, count, largest=False).indices.cpu().tolist())
            neighbors.update(goal_indices)
            lengths = distances.cpu().tolist()
            for destination in neighbors:
                length = lengths[destination]
                estimate = costs[source] + length
                if destination == source or length < 1e-8 or estimate >= costs[destination]:
                    continue
                if lower_cost[source] + length + heuristic[destination] >= best_cost:
                    pruned += 1
                    continue
                heapq.heappush(edge_queue, (estimate + heuristic[destination], source, destination, costs[source], length))
            neighbor_ms += (time.perf_counter() - neighbor_started) * 1000
        pending = []
        while edge_queue and len(pending) < edge_batch_size:
            priority, source, destination, queued_cost, length = heapq.heappop(edge_queue)
            if queued_cost != costs[source]:
                priority = costs[source] + length + heuristic[destination]
            if priority >= best_cost or costs[source] + length >= costs[destination]:
                continue
            pending.append((source, destination, length))
        if not pending:
            continue
        unseen = list(dict.fromkeys((min(source, destination), max(source, destination))
                                   for source, destination, _ in pending
                                   if (min(source, destination), max(source, destination)) not in edge_cache))
        if unseen:
            edge_started = time.perf_counter()
            source_indices, destination_indices = zip(*unseen)
            accepted, stability = validity.edges(nodes[list(source_indices)], nodes[list(destination_indices)],
                                                  return_stability=True)
            for pair, valid, penalty in zip(unseen, accepted.cpu().tolist(), stability.cpu().tolist()):
                edge_cache[pair] = (valid, penalty)
                edges_valid += int(valid)
            edges_checked += len(unseen)
            edge_ms += (time.perf_counter() - edge_started) * 1000
        for source, destination, length in pending:
            valid, penalty = edge_cache[min(source, destination), max(source, destination)]
            candidate_cost = costs[source] + length * (1 + validity.stability_weight * penalty)
            if not valid or candidate_cost >= costs[destination] or candidate_cost + heuristic[destination] >= best_cost:
                continue
            ancestor = source
            while ancestor >= 0 and ancestor != destination:
                ancestor = parents[ancestor]
            if ancestor == destination:
                continue
            previous_cost = costs[destination]
            if parents[destination] >= 0:
                children[parents[destination]].remove(destination)
                rewires += 1
            parents[destination] = source
            children[source].add(destination)
            costs[destination] = candidate_cost
            affected = [destination]
            for descendant in affected:
                if descendant != destination:
                    costs[descendant] += candidate_cost - previous_cost
                affected.extend(children[descendant])
                heapq.heappush(vertex_queue, (costs[descendant] + heuristic[descendant], descendant, costs[descendant]))
                record_solution(descendant)
    from v3_batched_loaded_search import densify

    checked = 0
    selected_path = selected_goal = selected_cost = None
    for cost, goal_index, path in sorted(solutions, key=lambda item: item[0]):
        checked += 1
        dense = densify(path, weights.cpu().tolist())
        if bool(validity.mask(torch.as_tensor(dense, device=start.device, dtype=start.dtype)).all().item()):
            selected_path, selected_goal, selected_cost = path, goal_index, cost
            break
    stats = dict(algorithm="BIT* GPU batch variant", batches=batches, tree_nodes=sum(math.isfinite(cost) for cost in costs),
                 sampled_nodes=len(nodes), edges_checked=edges_checked, edges_valid=edges_valid,
                 rewires=rewires, pruned_edges=pruned, first_solution_ms=first_solution_ms,
                 search_ms=(time.perf_counter() - started) * 1000, selected_goal=selected_goal,
                 goal_count=len(goals), best_weighted_length=selected_cost, fine_candidates_checked=checked,
                 raw_waypoints=len(selected_path) if selected_path is not None else 0,
                 sampling_ms=sampling_ms, neighbor_ms=neighbor_ms, edge_gpu_ms=edge_ms,
                 shortcut_applied=False, optimality_guarantee=False)
    return selected_path, stats

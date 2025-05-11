#include "bfs.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <climits>
#include <cstddef>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances) {
#pragma omp parallel
    {
        // need local frontier for each thread
        vertex_set local_frontier;
        vertex_set_init(&local_frontier, g->num_nodes);

// split work between threads
// dynamic better for uneven work
#pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < frontier->count; i++) {
            // get current vertex from frontier
            int current_vertex = frontier->vertices[i];

            // find where edges start and end
            int edge_start = g->outgoing_starts[current_vertex];
            int edge_end;

            // last vertex special case
            if (current_vertex == g->num_nodes - 1) {
                edge_end = g->num_edges;
            } else {
                edge_end = g->outgoing_starts[current_vertex + 1];
            }

            // check all neighbors
            int edge_idx = edge_start;
            while (edge_idx < edge_end) {
                int neighbor_vertex = g->outgoing_edges[edge_idx];

                // check if not visit yet
                if (distances[neighbor_vertex] == NOT_VISITED_MARKER) {
                    if (__sync_bool_compare_and_swap(&distances[neighbor_vertex],
                                                     NOT_VISITED_MARKER,
                                                     distances[current_vertex] + 1)) {
                        // add to local list if we got it first
                        local_frontier.vertices[local_frontier.count++] = neighbor_vertex;
                    }
                }

                edge_idx++;
            }
        }

        // merge
        if (local_frontier.count > 0) {
            // get space in global list atomically
            int insert_position = __sync_fetch_and_add(&new_frontier->count, local_frontier.count);

            for (int j = 0; j < local_frontier.count; j++) {
                new_frontier->vertices[insert_position + j] = local_frontier.vertices[j];
            }
        }

        free(local_frontier.vertices);
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {
        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

// each node check if it can be added to frontier from existing frontier nodes
void process_bottom_up(
    Graph g,
    vertex_set* frontier,
    vertex_set* next_frontier,
    int* distances) {
    bool* in_frontier = (bool*)calloc(g->num_nodes, sizeof(bool));

    int i = 0;
    while (i < frontier->count) {
        in_frontier[frontier->vertices[i]] = true;
        i++;
    }

#pragma omp parallel
    {
        // each thread need own frontier list
        vertex_set thread_frontier;
        vertex_set_init(&thread_frontier, g->num_nodes);

#pragma omp for schedule(dynamic, 256)
        for (int vertex = 0; vertex < g->num_nodes; vertex++) {
            if (distances[vertex] != NOT_VISITED_MARKER)
                continue;

            // find edges coming to this node
            int start_edge = g->incoming_starts[vertex];
            int end_edge;

            // handle last node special
            if (vertex == g->num_nodes - 1) {
                end_edge = g->num_edges;
            } else {
                end_edge = g->incoming_starts[vertex + 1];
            }

            // for tracking best parent
            bool found_parent = false;
            int distance_from_root = INT_MAX;

            // check all nodes
            int edge = start_edge;
            while (edge < end_edge && !found_parent) {
                int neighbor = g->incoming_edges[edge];

                if (in_frontier[neighbor]) {
                    int potential_distance = distances[neighbor] + 1;

                    // keep smallest distance
                    if (potential_distance < distance_from_root) {
                        distance_from_root = potential_distance;
                        found_parent = true;
                    }
                }

                edge++;
            }

            if (found_parent) {
                // try update my distance
                if (__sync_bool_compare_and_swap(&distances[vertex],
                                                 NOT_VISITED_MARKER,
                                                 distance_from_root)) {
                    thread_frontier.vertices[thread_frontier.count++] = vertex;
                }
            }
        }

        if (thread_frontier.count > 0) {
            // get spot to insert atomically
            int insert_position = __sync_fetch_and_add(&next_frontier->count, thread_frontier.count);

            // copy vertices to main list
            for (int j = 0; j < thread_frontier.count; j++) {
                next_frontier->vertices[insert_position + j] = thread_frontier.vertices[j];
            }
        }

        free(thread_frontier.vertices);
    }

    free(in_frontier);
}

void bfs_bottom_up(Graph graph, solution* sol) {
    // make two lists for switching
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    // rename for easier to understand
    vertex_set* current = &list1;
    vertex_set* next = &list2;

    // set all distance to not visit yet
    for (int i = 0; i < graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    // put root node in first list
    current->vertices[current->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    // loop until no more nodes to process
    while (current->count > 0) {
        // empty next list
        vertex_set_clear(next);

        // run bottom up step
        process_bottom_up(graph, current, next, sol->distances);

        // swap the lists
        vertex_set* temp = current;
        current = next;
        next = temp;
    }

    free(list1.vertices);
    free(list2.vertices);
}

void bfs_hybrid(Graph graph, solution* sol) {
    // need two list for swapping
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    // rename for less confusion
    vertex_set* current_frontier = &list1;
    vertex_set* next_frontier = &list2;

    // set all nodes to not visited
    int node_index = 0;
    while (node_index < graph->num_nodes) {
        sol->distances[node_index] = NOT_VISITED_MARKER;
        node_index++;
    }

    // start with only root node
    current_frontier->vertices[current_frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    // when to change between algorithms
    // can adjust these to make faster
    const double td_to_bu_threshold = 0.15;  // change to bottom when frontier big
    const double bu_to_td_threshold = 0.07;  // change to top when frontier small again

    // start with top down
    bool using_top_down = true;

    // loop until no more nodes
    while (current_frontier->count > 0) {
        // clear next list
        vertex_set_clear(next_frontier);

        // see how big frontier is compared to graph
        double frontier_ratio = (double)current_frontier->count / graph->num_nodes;

        // decide which algorithm better now
        if (using_top_down && frontier_ratio > td_to_bu_threshold) {
            // got big switch to bottom up
            using_top_down = false;
        } else if (!using_top_down && frontier_ratio < bu_to_td_threshold) {
            // got small go back to top down
            using_top_down = true;
        }

        // run the better algorithm for this step
        if (using_top_down) {
            top_down_step(graph, current_frontier, next_frontier, sol->distances);
        } else {
            process_bottom_up(graph, current_frontier, next_frontier, sol->distances);
        }

        // swap lists for next time
        vertex_set* temp = current_frontier;
        current_frontier = next_frontier;
        next_frontier = temp;
    }

    free(list1.vertices);
    free(list2.vertices);
}

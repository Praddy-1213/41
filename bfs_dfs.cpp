#include <iostream>
#include <queue>
#include <stack>
#include <omp.h>
#include <vector>
using namespace std;

struct Graph 
{
    int V;
    vector<vector<int>> adj;

    Graph(int V) 
    {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) 
    {
        adj[u].push_back(v);
        adj[v].push_back(u); // For undirected graph
    }

    void displayGraph() 
    {
        cout << "Graph representation (Adjacency List):\n";
        for (int i = 0; i < V; i++) {
            cout << i << ": ";
            for (int v : adj[i]) {
                cout << v << " ";
            }
            cout << endl;
        }
    }

    void BFS(int start) 
    {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        cout << "BFS traversal starting from node " << start << ": ";
        while (!q.empty()) 
        {
            int u = q.front();
            q.pop();
            cout << u << " ";

            #pragma omp parallel for
            for (int i = 0; i < adj[u].size(); i++) 
            {
                int v = adj[u][i];
                if (!visited[v]) 
                {
                    #pragma omp critical
                    {
                        if (!visited[v]) 
                        {
                            visited[v] = true;
                            q.push(v);
                        }
                    }
                }
            }
        }
        cout << endl;
    }

    void DFS(int start) 
    {
        vector<bool> visited(V, false);
        stack<int> s;

        s.push(start);
        visited[start] = true;

        cout << "DFS traversal starting from node " << start << ": ";
        while (!s.empty()) 
        {
            int u = s.top();
            s.pop();
            cout << u << " ";

            vector<int> neighbors;
            for (int v : adj[u]) 
            {
                if (!visited[v]) 
                {
                    neighbors.push_back(v);
                }
            }

            #pragma omp parallel for
            for (int i = 0; i < neighbors.size(); i++) 
            {
                int v = neighbors[i];
                #pragma omp critical
                {
                    if (!visited[v]) 
                    {
                        visited[v] = true;
                        s.push(v);
                    }
                }
            }
        }
        cout << endl;
    }
};

int main() 
{
    int V, edgeCount;

    cout << "Enter number of nodes: ";
    cin >> V;

    Graph g(V);

    cout << "Enter number of edges: ";
    cin >> edgeCount;

    cout << "Enter edges (start_node end_node): \n";
    for (int i = 0; i < edgeCount; i++) 
    {
        int u, v;
        cin >> u >> v;
        g.addEdge(u, v);
    }

    g.displayGraph();

    int root;
    cout << "Enter the root node of tree: ";
    cin >> root;

    cout << "BFS traversal from node ",root,":\n";
    g.BFS(root);

    cout << "DFS traversal from node ",root,":\n";
    g.DFS(root);

    return 0;
}
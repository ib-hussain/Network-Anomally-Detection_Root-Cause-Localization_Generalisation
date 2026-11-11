# Step 23 — THP to ACG graph compatibility

The exploratory gCastle THP output contained diagonal entries.

The ACG conditional-probability implementation explicitly
requires a directed graph with no self-loops.

Resolution:
- Preserve the original THP output unchanged.
- Create a separate graph with its diagonal set to zero.
- Preserve every off-diagonal edge.
- Record source and derived SHA-256 hashes.
- Calculate ACG weights using the derived graph.

Scientific limitations:
- The authors' self-loop policy is not confirmed.
- This is an explicit compatibility decision.
- The THP graph was learned from all Wireless records.
- ACG weights use an exploratory time-slicing protocol.
- The experiment is not a held-out TTHP reproduction.

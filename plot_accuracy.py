import json
import matplotlib.pyplot as plt
from pathlib import Path

json_path = Path("output/graph_enhancement_cache/csv_vs_graph_analysis/overall_summary.json")

with open(json_path, "r") as f:
    data = json.load(f)

comparisons = data["comparisons"]

combined_k = []
bl_plus_graph_acc = []
extended_bl_acc = []

for name, comp in comparisons.items():
    bl_k = comp["config"]["BL_k"]
    graph_k = comp["config"]["graph_k"]
    total_k = bl_k + graph_k
    combined_k.append(total_k)
    bl_plus_graph_acc.append(comp["BL_plus_graph_accuracy"])
    extended_bl_acc.append(comp["extended_BL_accuracy"])

# Sort by combined k
z = sorted(zip(combined_k, bl_plus_graph_acc, extended_bl_acc))
combined_k, bl_plus_graph_acc, extended_bl_acc = map(list, zip(*z))

plt.figure(figsize=(10, 6))
plt.plot(combined_k, bl_plus_graph_acc, marker='o', linewidth=2, markersize=8, label='BL + Graph (k1 + k2)')
plt.plot(combined_k, extended_bl_acc, marker='s', linewidth=2, markersize=8, label='Extended BL (k = k1 + k2)')
plt.xlabel('Total k (BL_k + graph_k)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy Comparison: BL + Graph vs Extended BL', fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=11)

# Add values on data points for better readability
for i, (k, bl_graph, ext_bl) in enumerate(zip(combined_k, bl_plus_graph_acc, extended_bl_acc)):
    plt.annotate(f'{bl_graph:.3f}', (k, bl_graph), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    plt.annotate(f'{ext_bl:.3f}', (k, ext_bl), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("output/graph_enhancement_cache/csv_vs_graph_analysis/accuracy_comparison.png", dpi=200, bbox_inches='tight')
plt.show()

# Print summary statistics
print(f"Data points: {len(combined_k)}")
print(f"K values: {combined_k}")
print(f"BL + Graph accuracies: {[f'{acc:.3f}' for acc in bl_plus_graph_acc]}")
print(f"Extended BL accuracies: {[f'{acc:.3f}' for acc in extended_bl_acc]}")

# Calculate improvement
improvements = [(bl_graph - ext_bl) for bl_graph, ext_bl in zip(bl_plus_graph_acc, extended_bl_acc)]
print(f"Improvements (BL+Graph - Extended BL): {[f'{imp:.3f}' for imp in improvements]}")
print(f"Average improvement: {sum(improvements)/len(improvements):.3f}")
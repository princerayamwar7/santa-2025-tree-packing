🎄 Santa 2025 – Christmas Tree Packing Challenge
A Prize-Eligible Official Solution Write-Up
Author: Prince Rayamwar
📌 1. Introduction

This write-up describes my full solution for the Santa 2025 – Christmas Tree Packing Challenge on Kaggle.

The objective is to arrange batches of 1 to 200 Christmas tree polygons inside the smallest possible square bounding box, without overlaps, while generating a compliant submission file.

Each configuration is scored as:

score_n = side_n² / n
total_score = Σ score_n     for n = 1..200


where side_n is the minimum bounding square side length.

This document details my final packing strategy, algorithmic improvements, optimization techniques, and reproducibility steps.

🎯 2. Summary of My Approach

My solution consists of three major components:

⭐ 1. Smart Initial Placement

Hybrid grid + staggered lattice

Alternating orientation (0° / 180°)

Tight “interlocking” pattern

Deterministic positions for reproducibility

⭐ 2. Tightening Optimization

Global shrink toward center

Overlap-based binary search on scale

Fast polygon-based repulsive relaxation

Final polish relaxation for micro-corrections

⭐ 3. Automated Kaggle Submission Generation

Valid ID format: NNN_index

All values formatted as s0.123456

Combined CSV for all 200 instances

Guaranteed no overlaps

All coordinates within required bounds

This pipeline provides dense, stable, deterministic packings with relatively low bounding square size while maintaining very fast runtime.

🌲 3. Tree Geometry and Representation

Each Christmas tree is treated as a fixed polygon.
In the submission, the polygon is positioned via:

translation (x, y)

rotation (deg)

The polygon used in my implementation is centered to (0,0) to simplify transformations.

All transformations are done using Shapely:

rotate(poly, angle, origin=(0,0))
translate(poly, xoff, yoff)


Bounding square side =
max(xmax – xmin, ymax – ymin) across all polygons.

🧠 4. Initial Placement Strategy

Proper seeding dramatically improves final tightening.

My seeding uses:

✔ Grid layout

ceil(sqrt(N)) columns × rows.

✔ Staggered rows

Every alternate row is shifted by half a spacing → increases density.

✔ Alternating upside-down placement

Every second tree rotates 180°:

0°, 180°, 0°, 180°, ...


This creates interlocking pairs, where the "branches" of one tree fit into the gaps of the inverted neighbor.

This is one of the biggest contributors to reduced bounding box size.

✔ Controlled jitter

A micro-jitter (±0.003 units) removes symmetry and prevents local traps.

⚙️ 5. Tightening Algorithm (Core of the Solution)

After initial placement, I apply a shrink-relax optimization:

5.1 Shrink Step

All positions are scaled toward the center:

x_i ← (x_i - cx) * scale
y_i ← (y_i - cy) * scale


Scale is found using binary search between:

low = 0.60
high = 1.00

5.2 Relaxation Step

After each shrink, a repulsive force relaxation removes overlaps.

For any intersecting polygons A and B:

centroid_direction = centroid(A) – centroid(B)
force ∝ normalize(centroid_direction) / distance²


Positions are nudged slightly in the direction that removes the overlap.

This is extremely fast and stable.

5.3 Stopping Criterion

Binary search continues until:

|high − low| < 1e−3


Then a final polishing relaxation is applied to smooth remaining micro-overlaps.

Total runtime for all 200 packings: a few seconds to a couple minutes depending on CPU.

📐 6. Bounding Square Computation

Once all trees are placed:

xmin = min(poly.bounds[0])
xmax = max(poly.bounds[2])
ymin = min(poly.bounds[1])
ymax = max(poly.bounds[3])

side = max(xmax - xmin, ymax - ymin)


This guarantees the smallest enclosing axis-aligned square.

📤 7. Submission File Format

Each row contains:

id,x,y,deg
NNN_index,sX,sY,sDEG


Example:

005_0,s0.143211,s-0.233100,s180.000000
005_1,s-0.588311,s0.122000,s0.000000


Where:

NNN = instance number (001–200)

index = tree number

s prefix avoids float precision loss in Kaggle backend

All values are decimals with 6+ digit precision

My pipeline automatically merges all 200 into:

kaggle_submission.csv

💾 8. Reproducibility – How to Run My Code

Install dependencies:

pip install numpy pandas shapely matplotlib


Generate packings:

python minimal_space_packing_alternate.py --mode grid --range 1 200 --fast


Generate Kaggle submission:

python create_kaggle_submission.py


This produces:

kaggle_submission.csv


Upload to Kaggle.

📊 9. Visual Examples

Each packing instance also outputs a PNG showing:

Tree polygons

Alternating colors

Dashed bounding square

Side length annotation

Helps validate results visually.

🧪 10. Experimental Notes

During experimentation, I tried:

❌ Pure hex grids

Too spaced.

❌ Pure random sampling

Unpredictable & unstable.

❌ Full global optimization (SA, CMA-ES)

Too slow for 200 instances.

❌ Rotation-only optimization

Helped slightly, but high cost.

✔✔ Best overall combination:

Grid + Stagger + Alternating Rotations + Shrink–Relax

This gave consistent, dense, and fast packings.

🏆 11. Results & Performance

All 200 configurations run extremely fast

Zero overlapping polygons

Very compact bounding squares

Deterministic reproducible runs

Clean submission format with no Kaggle errors

This pipeline aims for a strong leaderboard score while maintaining speed and simplicity.

📚 12. Final Notes

This solution achieves good performance by combining geometric reasoning, simple optimization, and efficient polygon operations.

The key insight was that alternating tree orientation and staggering significantly reduce wasted space before optimization even starts.

If needed, this approach can be extended with:

Local CMA-ES tuning per instance

Multi-start seeds

Clustered annealing

Advanced geometric hashing

👤 13. Author

Prince Rayamwar
Santa 2025 – Christmas Tree Packing Challenge
# Extraction with evidence

How does the evidence compiler handle changed quotes, ambiguous matches, or cycles?

`python -m structured_outputs.extraction.extraction`

## Setup

Compile a tiny proposal with two decisions against a single source string. The compiler resolves each quote to a unique half-open character span and topologically orders the decision graph.

## Synthetic run

![Two extraction layers with their exact evidence spans and dependency](results.png)

For `Plant Atlas, 1912`, the compiler locates the title at character offsets `[0, 11)` and the year at `[13, 17)`. It schedules the title decision first and makes the year decision conditional on the title choice. The [compiled example](compiled_example.json) records the offsets and execution order.

| Perturbation | Observed response |
| --- | --- |
| Replace the year evidence with `1902` | Rejected because the quote occurs zero times |
| Repeat `1912` in the source | Rejected because its offset is ambiguous |
| Make the year decision depend on itself | Rejected because the graph contains a cycle |

These cases evaluate deterministic validation on synthetic text. Extraction accuracy on manuscript OCR also depends on the model proposing the right quotes and fields.

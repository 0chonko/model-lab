# Extraction with evidence

How does the evidence compiler handle changed quotes, ambiguous matches, or cycles?

`python -m structured_outputs.extraction.extraction`

Change a quote, repeat it in the source, or introduce a cycle and inspect the compiler response.

## Synthetic run

For `Plant Atlas, 1912`, the compiler locates the title at character offsets `[0, 11)` and the year at `[13, 17)`. It schedules the title decision first and makes the year decision conditional on the title choice. The [compiled example](compiled_example.json) records the offsets and execution order.

try:
    from IPython.display import display, HTML

    def visualize_chunks(chunks, max_chunks=5, min_overlap=10):
        html_parts = []
        all_chunks = chunks
        chunks = chunks[:max_chunks]
        prev_overlap_len = 0
        for i, chunk in enumerate(chunks):
            overlap_len = 0
            is_cutoff_overlap = False
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                for length in range(1, min(len(chunk), len(next_chunk)) + 1):
                    if chunk.endswith(next_chunk[:length]):
                        overlap_len = length
            elif len(all_chunks) > max_chunks:
                next_chunk = all_chunks[max_chunks]
                for length in range(1, min(len(chunk), len(next_chunk)) + 1):
                    if chunk.endswith(next_chunk[:length]):
                        overlap_len = length
                if overlap_len > 0:
                    is_cutoff_overlap = True
            if overlap_len < min_overlap:
                overlap_len = 0
                is_cutoff_overlap = False

            html_parts.append(
                f"<h4>Chunk {i+1} ({len(chunk)} chars)</h4><pre style='white-space:pre-wrap;'>"
            )
            start = ""
            if prev_overlap_len > 0:
                overlap_start = chunk[:prev_overlap_len]
                start = f"<mark style='background:#b3e5fc;'>{overlap_start}</mark>"
                chunk = chunk[prev_overlap_len:]
            if overlap_len > 0:
                unique = chunk[:-overlap_len]
                overlap_end = chunk[-overlap_len:]
                html_parts.append(
                    f"{start}{unique}<mark style='background:#ffeb3b;'>{overlap_end}</mark>"
                )
                label = (
                    "overlap with next chunk (not shown)"
                    if is_cutoff_overlap
                    else "overlap with next chunk"
                )
                html_parts.append(
                    f"</pre><p><em>↕ {label}: {overlap_len} chars</em></p>"
                )
            else:
                html_parts.append(f"{start}{chunk}</pre>")
            prev_overlap_len = overlap_len

        display(HTML("".join(html_parts)))

except ImportError:

    def visualize_chunks(chunks, max_chunks=5, min_overlap=10):
        """Print chunks with overlapping portions highlighted in color."""
        RESET = "\033[0m"
        COLORS = [
            "\033[42m",  # green bg
            "\033[43m",  # yellow bg
            "\033[44m",  # blue bg
            "\033[45m",  # magenta bg
            "\033[46m",  # cyan bg
        ]

        all_chunks = chunks
        chunks = chunks[:max_chunks]

        prev_overlap_len = 0
        for i, chunk in enumerate(chunks):
            color = COLORS[i % len(COLORS)]
            print(f"\n{'='*60}")
            print(f"Chunk {i+1} (length: {len(chunk)})")
            print(f"{'='*60}")

            # Find overlap with next chunk
            overlap_len = 0
            is_cutoff_overlap = False
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                for length in range(1, min(len(chunk), len(next_chunk)) + 1):
                    if chunk.endswith(next_chunk[:length]):
                        overlap_len = length
            elif len(all_chunks) > max_chunks:
                next_chunk = all_chunks[max_chunks]
                for length in range(1, min(len(chunk), len(next_chunk)) + 1):
                    if chunk.endswith(next_chunk[:length]):
                        overlap_len = length
                if overlap_len > 0:
                    is_cutoff_overlap = True
            if overlap_len < min_overlap:
                overlap_len = 0
                is_cutoff_overlap = False

            start = ""
            if prev_overlap_len > 0:
                overlap_start = chunk[:prev_overlap_len]
                start = f"\033[46m{overlap_start}{RESET}"
                chunk = chunk[prev_overlap_len:]
            if overlap_len > 0:
                unique = chunk[:-overlap_len]
                overlap = chunk[-overlap_len:]
                print(f"{start}{unique}{color}{overlap}{RESET}")
                label = (
                    "overlap (with next chunk, not shown)"
                    if is_cutoff_overlap
                    else "overlap"
                )
                print(f"\n  ↕ {label}: {overlap_len} chars")
            else:
                print(f"{start}{chunk}")
            prev_overlap_len = overlap_len

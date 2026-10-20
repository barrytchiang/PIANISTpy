# The technical notes on the website

The technical notes are written in LaTeX (one file, `PIANISTpy_Technical_Notes.tex`) and published two ways: as a
PDF, and as pages of the manual under *API reference > Technical notes*. The pages are transcribed from the LaTeX by
`docs/tools/notes_pages.py`; nobody edits them by hand. This file is the procedure and the rules of that
transcription.

## Updating the pages after the notes change

1. Run the transcription on the current source:

   ```bash
   python docs/tools/notes_pages.py "<path>/PIANISTpy_Technical_Notes.tex"
   ```

   The tool compiles the notes twice with `pdflatex` in a temporary folder (nothing is written next to the source)
   and rewrites `docs/reference/notes.rst` and `docs/reference/technical-notes/*.rst`. With `--aux <file>` it
   reads the numbering from an existing compile instead.
2. Build the manual with warnings as errors (`python -m sphinx -W -b html docs <out>`).
3. Check the rendered pages: `python docs/tools/notes_check.py <out>`. It must report 0 findings.
4. Read the diff of the generated pages. A change there that the notes did not make is a transcription defect.
5. Look at two pages in a browser: the one with the most equations changed and one with a table. Equations
   render, every equation shows its tag at the right, cross references jump to their targets.
6. Commit the generated pages together with any change to the tool, in one commit whose message names the notes'
   version.
7. Delete a page of a section the notes removed, and its line goes from the toctree automatically. A renamed
   section label renames its page; delete the old page in the same commit.

## What the transcription guarantees

- **Numbers are LaTeX's own.** The tool counts sections, subsections, equations and tables itself and compares
  every labelled item with the `.aux` of the compile. A difference stops it before anything is written. All
  labels of the `.aux` must be met.
- **Nothing is dropped silently.** A command or environment the tool does not know stops it with the text around
  it. The notes may use only what the tool covers (below); a new construct is added to the tool first, in the same
  commit as the notes that need it.
- **One page per section.** Each `\section` is a page, named by its label (`tn:sec:df` is `df`); `\section*` (the
  references) is a page without a number. The page title is the section's number and title, which is also its
  menu label.
- **Anchors are the notes' labels.** `tn:eq:conditions` is the anchor `tn-eq-conditions` on its page, so an
  address can point at any equation, section or table.

## The rules of the transcription

| LaTeX in the notes | On the website |
|---|---|
| `\section`, `\subsection`, `\subsubsection` with `\label` | heading "T:2.3 Title", anchor from the label |
| `equation` with `\label` | displayed math with `\tag{T:2.4}`, anchor from the label |
| `$...$` | inline math (MathJax) |
| the notes' math macros (`\betac`, `\Ra`, `\Rt`, `\Kt`, `\K`, `\raux`, `\rmd`) | expanded to their definitions from the preamble |
| `\eref`, `\sref`, `\tref` | a link "Eq. (T:2.4)", "Section T:2.3", "Table T:2.1" to the anchor |
| `\paper`, `\peq`, `\psec`, `\papp`, `\pfig` | the site's link `|paper|` (Chiang+(2026d), set in `docs/conf.py`), then ", Eq. (B9)" and so on |
| `\peqn` | "Eq. (B9)" without a link (the paper was just named) |
| `\href{url}{text}`, `\citeX` macros | an external link (opens in a new tab on the site) |
| `\textit`, `\emph`, `\textbf` | italic, bold; math inside stays math (reST does not nest them) |
| `\PIANISTpy`, `\JAX`, `\galpy`, `\agama` | PIANISTpy, JAX, galpy, AGAMA |
| `itemize`, `enumerate`, `list` | bulleted and numbered lists; equations inside an item stay in the item |
| `table` with `tabular` and booktabs rules | a table titled "Table T:2.1. caption" |
| `~` | a non-breaking space |
| ` `` '' ` and `--` | curly quotes and a hyphen |
| `%` comments, `\maketitle`, `\tableofcontents`, `\noindent`, `\centering` | dropped |

The opening paragraph of the notes (before the first section) is the text of the *Technical notes* page, followed
by the version from `\date` and the sentence on how docstrings cite the notes.

## Writing the notes so they transcribe

- Every equation that a docstring or a page cites has a `\label{tn:eq:<key>}`; sections `tn:sec:<key>`, tables
  `tn:tab:<key>`. The key never changes once code cites it.
- Cite the paper only through `\paper`, `\peq` and the other macros, never by writing the authors; the site
  turns them into its one link.
- No figures, footnotes or `align` environments: they are not covered. A figure belongs to a notebook.
- Math macros are defined without arguments in the preamble and use only commands MathJax knows.
- A link's text holds no math.

## When the tool refuses

The message names the construct and the text around it. Either rewrite that place in the notes with a covered
construct, or extend the tool: add the case to `Notes._inline` (running text) or `Notes.flow` (blocks), add its
row to the table above, and rerun steps 1 to 5.

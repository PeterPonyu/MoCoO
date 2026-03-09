# Figure 1, Results Traceability, and Workflow Revision Prompt for Claude Code

Your task is to perform a rigorous review of the **article design**, **Figure 1**, the **experimental results pipeline**, and the **codebase workflow** so that the manuscript reaches a journal-ready standard and remains easy to update when new experiments are added.

## Core Objectives
Prioritize the following in order:

1. **Scientific correctness**
2. **Traceability and reproducibility**
3. **Alignment between manuscript, figures, tables, and code**
4. **Workflow maintainability for future experiment updates**
5. **Visual and structural polish**

Do not limit the review to code style or isolated plotting issues.  
You must verify that the manuscript design, figure architecture, result tables, and generated outputs are all mutually consistent and can be regenerated reliably from the current codebase.

---

## Mandatory Review Scope
Before proposing changes, inspect all of the following:

1. The manuscript structure and article design
2. All figures and tables related to the experimental results
3. The code used to generate Figure 1
4. The code used to generate result figures and result tables
5. The raw data, processed data, and intermediate artifacts
6. The actual rendered/exported figures and tables
7. The workflow that connects experiments to final manuscript outputs

You must review both:

- the **implementation path**  
  (raw experiment outputs → aggregation/processing → plotting/tabulation → exported manuscript assets), and
- the **final rendered outputs themselves**

Do not rely only on source code, file names, or data tables.  
You must actually inspect the rendered figure/table outputs and verify that they reflect the real project condition.

---

## High-Priority Tasks

### 1. Check the Article Design
Review whether the article design is coherent and correctly organized.

Specifically check:

- whether figure order in the manuscript matches the narrative flow,
- whether figure numbering and figure naming are fully aligned,
- whether table numbering and references are consistent,
- whether the current article structure may have caused misunderstandings about figure meaning,
- whether Figure 1 is placed and explained in the most logical position.

Important: the previous misunderstanding of Figure 1 architecture may have been caused by the fact that the **figure order in the article is not perfectly aligned with figure naming/order expectations**.  
You must explicitly account for this risk during review.

---

### 2. Verify All Experimental Results Are Traceable
All experimental results in figures and tables must be verifiable and traceable.

For every reported result, identify:

1. where the raw result comes from,
2. how it is processed,
3. which script or notebook transforms it,
4. which plotting or table-building script consumes it,
5. which final figure/table file contains it.

You must determine whether every displayed result can be traced back to a reproducible source.

Flag any case where:

- a result cannot be traced,
- a figure/table appears manually edited,
- aggregation logic is unclear,
- naming is inconsistent,
- the manuscript claims are not directly supported by reproducible artifacts.

---

### 3. Ensure Future Experiment Updates Are Easy
The codebase should support easy refresh of figures and tables when new or enhanced experiments are performed.

Evaluate whether the current workflow allows newly generated results to be incorporated with minimal manual work.

Specifically check whether:

- new experiment outputs can be dropped into a clear location,
- aggregation scripts automatically pick them up,
- figures and tables refresh without manual patching,
- file naming conventions are stable,
- result schemas are consistent across experiments,
- the workflow avoids hard-coded values and one-off manual edits.

If the current workflow is not update-friendly, propose a revised pipeline and file organization.

The final recommendation should make it easy to:

- rerun experiments,
- regenerate summary results,
- refresh Figure 1 and all result figures,
- refresh result tables,
- audit the provenance of every reported value.

---

### 4. Re-evaluate Figure 1 Architecture Carefully
You must carefully read the **actual Figure 1 architecture** and verify that it correctly reflects the project’s real method, workflow, and current implementation status.

Do not assume the architecture is correct based only on:

- code comments,
- old descriptions,
- figure labels,
- manuscript references,
- earlier interpretations.

You must cross-check Figure 1 against:

- the manuscript description,
- the implemented pipeline,
- the experimental workflow,
- actual data/model flow in the repository,
- current project condition.

Important:  
You previously misunderstood the Figure 1 architecture.  
Therefore, this time you must explicitly guard against misinterpretation caused by:

- misleading figure order,
- mismatch between figure names and manuscript order,
- outdated assumptions,
- architecture diagrams that are simplified but not fully faithful.

Flag all mismatches, including:

- incorrect module ordering,
- missing steps,
- wrong arrows or dependencies,
- inconsistent terminology,
- unsupported architectural claims,
- visual structure that does not match the codebase workflow.

---

### 5. Inspect the Rendered Figure 1 Itself
You must inspect the actual rendered Figure 1, not only the source code and source data.

This is mandatory.

Check for visible issues such as:

- excess white space,
- incorrect panel balance,
- misaligned elements,
- clipped text,
- overlapping labels,
- unreadable annotations,
- inconsistent font sizes,
- poor export quality,
- strange legend artifacts,
- architecture layout issues,
- residual problems not obvious from code inspection.

You must read the figure as a reader would, not only as a developer would.

---

### 6. Reduce White Space and Fix Legend Issues
Figure 1 should be improved to publication standard.

#### White Space
Identify and reduce unnecessary white space caused by:

- excessive outer margins,
- oversized panel gaps,
- poor aspect ratio choices,
- legends placed too far from content,
- empty architecture regions,
- misaligned subpanel boundaries.

The figure should become more compact without harming readability.

#### Legend Defect
The legends currently show a strange **top horizontal line**.  
Determine whether it is caused by:

- legend frame settings,
- axis spines,
- table-like styling,
- separator drawing,
- export artifacts,
- panel overlay behavior.

Fix the issue and ensure the legends are clean and visually consistent.

---

## Workflow and Architecture Revision Requirement
Based on the investigation, determine whether the **current codebase architecture or workflow pipeline should be modified**.

If modification is needed, provide:

1. the current workflow problems,
2. the risks they cause,
3. the revised workflow design,
4. the expected benefits for traceability and maintainability,
5. the specific code/data organization changes required.

Focus especially on whether the figure/table generation system should be reorganized so that future experiment updates automatically propagate into manuscript-ready outputs.

---

## Expected Output
Your response must include:

### A. Diagnosis
A concise but concrete diagnosis of:

- article design issues,
- Figure 1 issues,
- architecture understanding issues,
- result traceability issues,
- workflow update problems,
- reproducibility weaknesses.

### B. Verification
Explicit verification of:

- whether Figure 1 architecture matches the actual project condition,
- whether all experimental results are traceable,
- whether figures and tables can be refreshed easily after new experiments,
- whether the rendered Figure 1 contains unresolved visual defects.

### C. Revision Plan
A prioritized plan covering:

1. manuscript/article design fixes,
2. Figure 1 architecture corrections,
3. whitespace reduction,
4. legend top-line removal,
5. result traceability improvements,
6. figure/table regeneration pipeline improvements,
7. codebase workflow restructuring if needed.

### D. Concrete Implementation Guidance
Provide actionable next steps, including:

- which files/scripts should be inspected or revised,
- which parts of the pipeline should be refactored,
- how to enforce traceable result provenance,
- how to make future experiment updates refresh figures/tables automatically.

---

## Operating Rules
- Do not assume figure order implies conceptual order.
- Do not assume figure names reflect actual manuscript sequence correctly.
- Do not assume Figure 1 architecture is correct without cross-checking the repository and manuscript.
- Do not assume tables and figures are traceable unless the full pipeline is verified.
- Do not stop at code inspection; inspect rendered outputs directly.
- Prefer code-driven regeneration over manual editing.
- Prioritize correctness and reproducibility over cosmetic changes.

---

## Final Priority Rule
Always prioritize:

**scientific correctness > traceability > reproducibility > updateability > visual polish**

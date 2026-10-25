# AI assistance log

Author: Tan Xuan Ying (34478027)
Unit: MMA3001, Project Dataset 1 (RGB-D Apple-Tree Reconstruction and Orchard Mapping)

## Tools used

Claude (Anthropic), used through a chat and coding-agent interface in October 2026.

## What the AI did

- Explained the project brief and the dataset, and proposed the task definition: predicting a 2D trunk and branch mask from a single RGB-D image, checked against the supplied ground-truth masks.
- Wrote the first versions of the notebook cells I ran in Colab.
- Wrote the `appleskel` package, the tests (`tests/`) and the experiment scripts (`scripts/`).
- Wrote the first drafts of the report sections and the README.

## What I did myself

- Created the GitHub repository and uploaded the code in separate commits, each with its own message.
- Ran the notebook (Steps 1 to 9) in Colab on the real dataset, and fixed the problems that appeared, such as file paths and missing variables after the session reset.
- Ran the test suite in Colab (40 tests passed) and generated the documentation pages with pdoc.
- Read and checked the results, then chose which results and settings to report.
- Edited the report, shortened it, added the references and links, and exported the final PDF.
- Decided to report a modest result honestly, and chose ridge regression over the U-Net because it gave the same score more consistently and cost less.

## Where the AI was wrong or overclaimed

- It first suggested multi-view ICP alignment before the data had been inspected. The data has only one view per tree, so the idea was dropped.
- It reported a ridge gain of +0.03 from one split. Repeating the experiment over 10 splits gave about +0.02, and part of the first figure came from the threshold grid.
- It first described 3D reconstruction as out of scope, then corrected this to an unvalidated extension.
- It said teaching-team approval was needed for the project. That rule applies only to student-proposed projects.
- A regression test assumed 23 orchard rows, but the data has 20. Running the test showed the error.
- A test used the wrong tensor dimension (`flip(2)` instead of `flip(-1)`). My first Colab run showed it, and it was fixed.
- The U-Net code was written without being able to train PyTorch in the AI's own environment, so it needs to be run and checked by me.

## How I checked the AI's output

- I ran the 40 unit tests in Colab. They include checks that the dual ridge solution matches the primal solution, that the SVD shape model matches scikit-learn PCA, that the metrics match hand-calculated examples, and that no orchard row appears in two splits.
- I ran the experiment on 10 different row-wise splits instead of trusting one split.
- I compared the numbers in the report with the outputs printed in my own notebook.

## What I would do differently

- Ask the AI to explain each step before running its code, so that I understand it earlier.
- Run the U-Net on several seeds and splits, so it can be compared fairly with ridge.
- Start the report and the repository earlier, and keep the report within the page limit.

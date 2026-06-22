# jamessteiner.github.io

My personal site for writeups of projects in robot learning and machine learning.

**Live at [jamessteiner.github.io](https://jamessteiner.github.io)**

## Writeups

- **[Flow vs. regression vs. diffusion: VLA action heads](https://jamessteiner.github.io/projects/vla-action-heads/)**: a controlled comparison of three action objectives (flow matching, L1 regression, DDPM diffusion) in a Vision-Language-Action model, holding everything else fixed.
- **[Can a Transformer Learn Position in a Few Thousand Parameters?](https://jamessteiner.github.io/projects/periodic-attention-bias/)**: replacing nanoGPT's position-embedding table with a ~3,500-parameter periodic attention bias on enwik8.

Built with [Jekyll](https://jekyllrb.com/), hosted on GitHub Pages.

<details>
<summary>Local development</summary>

```bash
bundle install
bundle exec jekyll serve   # http://localhost:4000
```

Add a writeup: drop a markdown file in `_projects/` with front matter
(`layout` / `title` / `summary` / `thumbnail` / `repo` / `date`); the home page
lists projects newest-first by `date`, and figures go in `assets/img/`.
</details>

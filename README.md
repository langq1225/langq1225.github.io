# langq1225.github.io

Personal academic website of Langqing Cui — <https://langq1225.github.io>

Built with [Jekyll](https://jekyllrb.com/) and the [al-folio](https://github.com/alshedivat/al-folio) theme.

## Where the content lives

| What                                | File                       |
| ----------------------------------- | -------------------------- |
| Bio, profile photo, homepage layout | `_pages/about.md`          |
| News items on the homepage          | `_news/*.md`               |
| Publications                        | `_bibliography/papers.bib` |
| Name, site URL, feature flags       | `_config.yml`              |
| Social links                        | `_data/socials.yml`        |
| Co-author homepage links            | `_data/coauthors.yml`      |
| CV data                             | `_data/cv.yml`             |
| Images                              | `assets/img/`              |

## Pages kept but not published

`blog`, `cv`, `projects`, `teaching`, `books`, `repositories` and `people` are present as
templates with `published: false` in their front matter. To turn one back on, set
`published: true` and `nav: true`, and re-enable whatever it needs in `_config.yml`
(the `collections` block for books/projects/teachings, `pagination` and
`jekyll-archives.posts` for the blog).

## Local preview

Requires Ruby ≥ 3.0 and ImageMagick (`brew install ruby imagemagick`).

```bash
bundle install
bundle exec jekyll serve
```

Then open <http://localhost:4000>.

## Deployment

Pushing to `main` triggers `.github/workflows/deploy.yml`, which builds the site and
pushes `_site/` to the `gh-pages` branch. GitHub Pages for this repository must be set to
**Deploy from a branch → `gh-pages` / `(root)`** (Settings → Pages).

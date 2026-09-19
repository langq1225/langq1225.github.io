# langq1225.github.io

Personal academic website of Langqing Cui — <https://langq1225.github.io>

Built with [Jekyll](https://jekyllrb.com/) and the [al-folio](https://github.com/alshedivat/al-folio) theme.

## Updating the site

Edit a file, commit, push to `main`. That's it — `.github/workflows/deploy.yml` builds the
site and publishes it to the `gh-pages` branch, which GitHub Pages serves. Takes ~2 minutes.

| To change                           | Edit                                                                                             |
| ----------------------------------- | ------------------------------------------------------------------------------------------------ |
| Bio / profile photo                 | `_pages/about.md`                                                                                |
| News on the homepage                | add a file to `_news/` (copy an existing one)                                                    |
| Publications                        | `_bibliography/papers.bib`                                                                       |
| Publication thumbnail               | drop an image in `assets/img/publication_preview/`, then add `preview = {file.png}` to the entry |
| Email / GitHub / Scholar / LinkedIn | `_data/socials.yml`                                                                              |
| CV contents                         | `_data/cv.yml`                                                                                   |
| Theme colour, feature flags         | `_config.yml`, `assets/css/main.scss`                                                            |

Adding a paper: append an entry to `_bibliography/papers.bib`. `selected = {true}` also puts
it on the homepage. Useful fields: `pdf`, `code`, `website` (renders as "Project Page"),
`abbr` (the coloured badge — define its colour in `_data/venues.yml`), `preview`,
`hide_date = {true}` (drops the date from the venue line).

## Local preview

```bash
export PATH="/opt/homebrew/opt/ruby/bin:$PATH"   # Homebrew Ruby; macOS system Ruby is too old
bundle exec jekyll serve
```

Then open <http://localhost:4000>. First time only: `brew install ruby imagemagick && bundle install`.

## Pages kept but not published

`Blog`, `CV`, `Projects`, `Teaching`, `Bookshelf`, `Repositories` and `People` exist as
templates with `published: false`. To enable one, set `published: true` and `nav: true`, and
re-enable what it needs in `_config.yml` — the `collections` block for
books/projects/teachings, `pagination` and `jekyll-archives.posts` for the blog.

## Local overrides of the theme

The theme ships as a gem, but local files win. These copies exist so they can be customised;
when upgrading al-folio, diff them against the gem:

- `_layouts/about.liquid` — photo above the header, socials in the navbar, capitalised headings
- `_layouts/bib.liquid` — "Website" button relabelled "Project Page", `hide_date` support
- `_includes/header.liquid` — full name bold in the navbar
- `assets/css/main.scss` — KAIST-blue theme colour and small layout tweaks, appended at the end

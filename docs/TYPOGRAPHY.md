# Typography

Direction: a modern, readable academic site with a quiet hierarchy. Optimistic
AI VF gives names and titles their character; Inter keeps body text and metadata
clear and consistent. Chinese uses the system sans-serif font. Size, weight,
spacing and contrast provide the hierarchy within each family.

The rules live in `_sass/_site-typography.scss`, loaded after the theme styles.
Inter, including its true italic, is loaded through `_config.yml` under
`third_party_libraries.google_fonts`. Optimistic AI VF is declared with
`@font-face` in the typography partial, using the font resource from
[Meta's official stylesheet](https://static.xx.fbcdn.net/rsrc.php/v5/yl/l/0,cross/fp-xVv2THvS.css).
It prefers an installed copy and falls back to Inter if the hosted font is unavailable.
Use `--font-heading` for headings and `--font-sans` for body text.

| Role                                     | Desktop size | Weight  | Line height        | Treatment                                                       |
| ---------------------------------------- | ------------ | ------- | ------------------ | --------------------------------------------------------------- |
| Name / page title                        | 36px         | 700     | 1.2                | Optimistic AI VF; darkest text; slight negative tracking        |
| Section title                            | 24px         | 600     | 1.35               | Optimistic AI VF; retain fine section dividers                  |
| Bio / body                               | 16px         | 400     | 1.75 in paragraphs | Primary text; emphasis at 600                                   |
| Paper title                              | 17px         | 600     | 1.5                | Optimistic AI VF; primary scan target; slight negative tracking |
| News / authors                           | 15px         | 400     | 1.65–1.7           | Authors use secondary color; own name at 600                    |
| Venue / dates / year groups / navigation | 14px         | 400–500 | 1.5–1.7            | Inter; venue name is italic; dates use tabular numerals         |
| Footer                                   | 13px         | 400     | 1.6                | Muted, without fading to low contrast                           |
| Notes / paper buttons / badges           | 12px         | 400–600 | 1.3–1.6            | Supporting information only                                     |

Mobile keeps body text at 16px; the name drops to 32px and section titles to 22px.
Sizes use rem units so browser font preferences remain effective.

## Color and emphasis

| Role                  | Light theme | Dark theme |
| --------------------- | ----------- | ---------- |
| Headings              | `#171c26`   | `#f3f5f9`  |
| Body                  | `#303640`   | `#d6dbe5`  |
| Secondary information | `#596273`   | `#a8b1c2`  |
| Notes / footer        | `#6b7280`   | `#939eaf`  |

Keep the site's existing blue for links and publication highlights. Use 600 for
emphasis within text and reserve 700 for the name. Venue names use Inter italic
to distinguish them from the author list. Keep distinctions such as "Oral
Presentation" upright, blue and semibold; avoid italicizing the whole row.
Prefer a concise venue name where possible. Avoid all-caps labels, widely spaced
body text and very light font weights.

Keep the default monospace styling for code. Do not apply the body font to icon
classes. When adding content, assign its semantic role instead of introducing a
new font or an arbitrary size.

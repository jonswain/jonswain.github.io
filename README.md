# jonswain.github.io

A Jekyll-powered personal blog focused on data science, cheminformatics, and applied machine learning in drug discovery. The site uses the Minima theme with custom layouts, SEO enhancements, topic landing pages, redirect validation, and post-level sharing/reading UX improvements.

## Project structure

```text
jonswain.github.io/
├── _config.yml                 # Site settings, plugins, permalink strategy
├── _includes/                  # Shared partials (head, header, footer, scripts)
├── _layouts/                   # Page/post/home layout templates
├── _posts/                     # Blog posts (Markdown with front matter)
├── _sass/minima/               # Custom styles and Minima overrides
├── assets/                     # Compiled/scss and static assets
├── images/                     # Post images and media
├── topics/                     # Topic landing pages
├── scripts/
│   └── check_redirect_canonicals.rb  # Redirect canonical QA check
├── README.md                   # Project documentation
└── Gemfile                     # Ruby/Jekyll dependencies
```

## Local development

Install dependencies:

```zsh
bundle install
```

To locally host the blog:

```zsh
bundle exec jekyll serve --livereload
```

To run a production-style build:

```zsh
bundle exec jekyll build
```

To create a .md version of a .ipynb file for a post:

```zsh
jupyter nbconvert --to markdown post.ipynb
```

## Post front matter

Typical post keys used in this repo:

```yaml
layout: post
title: "My Post Title"
date: 2026-01-01 12:00:00 +0000
permalink: /my-post-slug/
description: "Short SEO description"
categories:
  - data-science
```

Optional keys:

```yaml
author: Jon Swain
modified_date: 2026-01-02 12:00:00 +0000
preview: "Optional short teaser text"

# Explicit share image. String, or a hash with `path` (share) + `card` (homepage thumbnail).
image:
  path: /images/my_post/social-preview.png
  card: /images/my_post/card-preview.png
  alt: "Alt text for the homepage card thumbnail"

# Or just a string, used for both the share image and (via jekyll-feed) the RSS thumbnail:
# image: /images/my_post/preview.png

redirect_from:
  - /old/path/to/post.html
series: my-series
series_order: 2
```

### Card metadata behavior

- Homepage / topic-page cards only render a thumbnail when `image.card` is defined.
- No fallback image is used for cards.
- Homepage card summary text resolves in this order:
  1. `preview`
  2. `excerpt` (generated from post content)

## SEO notes

Page `<head>` metadata (title, description, canonical, Open Graph, JSON-LD) is produced
by `jekyll-seo-tag`. It reads the `image:` key (a string, or a hash with `path:`) for the
`og:image` / `twitter:image` tags.

`_includes/head.html` adds a fallback for pages that do **not** set `image:`. The share
image resolves in this order:

1. `image` string / `image.path` — handled by `jekyll-seo-tag`
2. `image.card`
3. first image found in the post body (`![](...)` or `<img src="...">`)
4. `/images/taranaki.jpg` (site default)

Note: `twitter:card` is `summary_large_image` only when `jekyll-seo-tag` resolves the
image itself (tier 1). Tiers 2–4 still emit `og:image` / `twitter:image`, but
`jekyll-seo-tag` hard-codes `twitter:card: summary` for them, and `jekyll-feed` only
emits an RSS `<media:thumbnail>` for tier 1. Set an explicit `image:` on posts you want
to promote for the large X card and a feed thumbnail.

## Stylesheet and formatting notes

- Site CSS is linked from `/assets/main.css`.
- The active Sass entrypoint is `assets/main.scss`.
- To prevent formatter-on-save issues with Sass files, workspace settings disable format-on-save for css/scss/sass in `.vscode/settings.json`.
- `.prettierignore` also excludes:
  - `assets/main.scss`

To verify redirect pages include canonical tags to their target permalink:

```zsh
bundle exec jekyll build && ruby scripts/check_redirect_canonicals.rb
```

## Sitemaps

- Main sitemap: `/sitemap.xml` (via `jekyll-sitemap`) — includes the topic landing pages.

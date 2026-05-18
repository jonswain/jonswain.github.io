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
├── topics-sitemap.xml          # Topic-specific sitemap
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
social_image: /images/my_post/social-preview.png
preview: "Optional short teaser text"

image:
  src: /images/my_post/base-image.png
  alt: "Optional alt text for homepage card image"
  social: /images/my_post/social-preview.png
  card: /images/my_post/card-preview.png

# Legacy form still supported for social fallback:
# image: /images/my_post/fallback-image.png

redirect_from:
  - /old/path/to/post.html
series: my-series
series_order: 2
```

### Card metadata behavior

- Homepage cards only render an image when `image.card` is defined.
- No fallback image is used for cards.
- Homepage card summary text resolves in this order:
  1. `preview`
  2. `excerpt` (generated from post content)

## SEO notes

Posts support structured image metadata and optional `social_image` for share previews:

```yaml
social_image: /images/my_post/social-preview.png
```

Social preview image fallback order in templates is:

1. `social_image`
2. `image.social`
3. `image.card`
4. `image.src` (or legacy `image` string)
5. `images.defaults.social` from `_config.yml`
6. `images.defaults.card` from `_config.yml`
7. `site.image` (if configured)

When a social image resolves, Open Graph and Twitter image tags are emitted.

Global image defaults are configured in `_config.yml`:

```yaml
images:
  defaults:
    social: /images/taranaki.jpg
    card: /images/taranaki.jpg
    hero: /images/taranaki.jpg
```

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

- Main sitemap: `/sitemap.xml` (via `jekyll-sitemap`)
- Topic-only sitemap: `/topics-sitemap.xml`

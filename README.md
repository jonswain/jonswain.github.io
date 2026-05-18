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
image: /images/my_post/fallback-image.png
redirect_from:
  - /old/path/to/post.html
series: my-series
series_order: 2
```

## SEO notes

Posts support an optional `social_image` front matter key for explicit share previews:

```yaml
social_image: /images/my_post/social-preview.png
```

If `social_image` is omitted, the site falls back to `image` from front matter/defaults.
If neither is present, no explicit image tags are rendered by the custom head logic.

To verify redirect pages include canonical tags to their target permalink:

```zsh
bundle exec jekyll build && ruby scripts/check_redirect_canonicals.rb
```

## Sitemaps

- Main sitemap: `/sitemap.xml` (via `jekyll-sitemap`)
- Topic-only sitemap: `/topics-sitemap.xml`

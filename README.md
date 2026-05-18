# jonswain.github.io

A personal blog on data science and cheminformatics.

## Local development

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

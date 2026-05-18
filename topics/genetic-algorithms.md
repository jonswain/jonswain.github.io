---
layout: page
title: Genetic Algorithms
permalink: /topics/genetic-algorithms/
---

{%- assign topic_posts = site.posts | where_exp: "post", "post.categories contains 'genetic-algorithms'" -%}

{%- if topic_posts.size > 0 -%}
<ul class="post-list">
  {%- for post in topic_posts -%}
  <li>
    <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>
    <h3><a class="post-link" href="{{ post.url | relative_url }}">{{ post.title | escape }}</a></h3>
    <p class="post-summary">{{ post.excerpt | strip_html | normalize_whitespace | truncate: 240 }}</p>
    <p><a class="read-more" href="{{ post.url | relative_url }}">Read article</a></p>
  </li>
  {%- endfor -%}
</ul>
{%- else -%}
<p>No posts found for this topic yet.</p>
{%- endif -%}

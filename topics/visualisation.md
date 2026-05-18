---
layout: page
title: Visualisation
permalink: /topics/visualisation/
---

{%- assign topic_posts = site.posts | where_exp: "post", "post.categories contains 'visualisation'" -%}

{%- if topic_posts.size > 0 -%}
<ul class="post-list">
  {%- assign date_format = site.minima.date_format | default: "%b %-d, %Y" -%}
  {%- for post in topic_posts -%}
  {%- include post_card.html post=post date_format=date_format show_summary=site.show_excerpts -%}
  {%- endfor -%}
</ul>
{%- else -%}
<p>No posts found for this topic yet.</p>
{%- endif -%}

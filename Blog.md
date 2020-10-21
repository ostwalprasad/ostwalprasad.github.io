---
title: Blog
permalink: "/Blog/"
layout: page
---

<div >




  {%- if site.posts.size > 0 -%}
    <ul >
      {%- for post in site.posts -%}
      <li>
        {%- assign date_format = site.minima.date_format | default: "%b %-d, %Y" -%}
        <span class="post-meta">{{ post.date | date: date_format }}</span>
        <h2>
          <a class="post-link" href="{{ post.url | relative_url }}">
            {{ post.title | escape }}
          </a>
        </h2>
        {%- if site.show_excerpts -%}
          {{ post.excerpt }}
        {%- endif -%}
      </li>
      {%- endfor -%}
    </ul>

  {%- endif -%}

  <!-- Icons -->
  <link rel="apple-touch-icon-precomposed" sizes="144x144" href="{{ site.baseurl }}public/apple-touch-icon-144-precomposed.png"><link rel="shortcut icon" href="{{ site.baseurl }}public/favicon.ico">

</div>

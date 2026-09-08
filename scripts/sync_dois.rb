#!/usr/bin/env ruby
# frozen_string_literal: true

# Sync Rogue Scholar DOIs into post front matter.
#
# Fetches this blog's post list from the Rogue Scholar API and writes a
# `doi:` key into the matching file in _posts/. Idempotent: only files
# whose DOI is missing or changed are rewritten. Run it after Rogue
# Scholar has registered DOIs for newly published posts, then commit.
#
#   ruby scripts/sync_dois.rb

require "json"
require "net/http"
require "uri"

ROOT = File.expand_path("..", __dir__)
POSTS_DIR = File.join(ROOT, "_posts")
BLOG_SLUG = "jonswain"
API_URL = "https://api.rogue-scholar.org/blogs/#{BLOG_SLUG}"

def fetch_posts
  body = Net::HTTP.get(URI(API_URL))
  JSON.parse(body).fetch("posts", [])
rescue StandardError => e
  abort "Failed to fetch #{API_URL}: #{e.message}"
end

def bare_doi(doi)
  doi.to_s.sub(%r{\Ahttps?://doi\.org/}, "").strip
end

# Rogue Scholar's `guid` is the feed entry id, whose last path segment is
# the Jekyll post slug (the filename with the date prefix stripped).
def slug_from_guid(guid)
  URI.parse(guid.to_s).path.split("/").reject(&:empty?).last
rescue URI::InvalidURIError
  nil
end

def post_file(slug)
  return nil unless slug

  Dir.glob(File.join(POSTS_DIR, "*-#{slug}.md")).find do |path|
    File.basename(path) =~ /\A\d{4}-\d{2}-\d{2}-#{Regexp.escape(slug)}\.md\z/
  end
end

# Insert or update the `doi:` line without disturbing the rest of the
# front matter (a YAML round-trip would drop the block list styling).
def apply_doi(path, doi)
  text = File.read(path)
  fm_end = text.start_with?("---\n") ? text.index("\n---\n", 4) : nil
  unless fm_end
    warn "  ! #{File.basename(path)}: no front matter, skipped"
    return false
  end

  front = text[0...fm_end]
  rest = text[fm_end..]
  current = front[/^doi:\s*(.+)$/, 1]&.strip

  return false if current == doi

  if current
    front = front.sub(/^doi:.*$/, "doi: #{doi}")
  else
    anchor = front[/^permalink:.*\n/] || front[/^date:.*\n/] || front[/^title:.*\n/]
    front = front.sub(anchor, "#{anchor}doi: #{doi}\n")
  end

  File.write(path, front + rest)
  true
end

changed = 0
unmatched = []

fetch_posts.each do |post|
  doi = bare_doi(post["doi"])
  next if doi.empty?

  slug = slug_from_guid(post["guid"])
  path = post_file(slug)
  if path.nil?
    unmatched << [post["title"], slug]
    next
  end

  next unless apply_doi(path, doi)

  puts "  + #{File.basename(path)} -> #{doi}"
  changed += 1
end

puts changed.zero? ? "All DOIs already up to date." : "Updated #{changed} post(s)."

unless unmatched.empty?
  puts "\nNo local file matched:"
  unmatched.each { |title, slug| puts "  - #{title.inspect} (slug: #{slug.inspect})" }
end

#!/usr/bin/env ruby
# frozen_string_literal: true

require "yaml"
require "date"
require "cgi"
require "uri"

ROOT = File.expand_path("..", __dir__)
SITE_DIR = File.join(ROOT, "_site")

def parse_frontmatter(path)
  text = File.read(path)
  match = text.match(/\A---\n(.*?)\n---\n/m)
  return {} unless match

  YAML.safe_load(match[1], permitted_classes: [Time, Date], aliases: true) || {}
end

def canonical_path(url)
  URI.parse(url).path
rescue URI::InvalidURIError
  nil
end

def check_file(path, expected_permalink)
  return [false, "missing generated redirect file"] unless File.exist?(path)

  content = File.read(path)
  canonical = content.match(/<link rel=\"canonical\" href=\"([^\"]+)\">/)&.captures&.first
  return [false, "missing canonical tag"] unless canonical

  return [true, nil] if canonical_path(canonical) == expected_permalink

  [false, "canonical mismatch (got #{canonical})"]
end

errors = []

Dir[File.join(ROOT, "_posts", "*.md")].sort.each do |post_file|
  fm = parse_frontmatter(post_file)
  next unless fm["redirect_from"]

  redirects = Array(fm["redirect_from"]).map(&:to_s)
  permalink = fm["permalink"].to_s
  next if permalink.empty?

  expected_permalink = permalink

  redirects.each do |redirect|
    rel = CGI.unescape(redirect.sub(%r{^/}, ""))
    site_path = File.join(SITE_DIR, rel)

    ok, message = check_file(site_path, expected_permalink)
    next if ok

    errors << "#{site_path}: #{message}"
  end
end

if errors.empty?
  puts "All redirect pages contain canonical tags pointing to their target permalinks."
  exit 0
end

puts "Redirect canonical check failed:"
errors.each { |e| puts "- #{e}" }
exit 1

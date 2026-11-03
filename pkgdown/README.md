Run `pkgdown::build_site()` to build the site locally. With the current
development version and `mode: auto`, it writes to `docs/dev`. Use
`pkgdown::preview_site()` to serve the built site, then `pkgdown::stop_preview()`
to stop the server. Previewing requires the optional `nanonext` package; install
it if prompted.

CI publishes the latest CRAN release at the site root and the current development
version under `/dev/`. The release pages use the matching tagged package source
with the current site configuration and assets. If a GitHub release runs before
CRAN accepts it, rerun the workflow manually after acceptance to update the root
site.

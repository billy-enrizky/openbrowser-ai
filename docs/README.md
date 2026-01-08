# OpenBrowser Documentation

This directory contains the documentation for OpenBrowser.

## Documentation Hosting

The documentation is built using [Mintlify](https://mintlify.com/) and can be hosted in several ways:

### Option 1: Mintlify Hosting (Recommended)

1. Create an account at [mintlify.com](https://mintlify.com/)
2. Connect your GitHub repository
3. Point to the `/docs` directory
4. Your docs will be available at your custom domain

### Option 2: GitHub Pages with Docusaurus

To host on GitHub Pages at `openbrowser.github.io/docs`:

1. We've included a GitHub Actions workflow at `.github/workflows/docs.yml`
2. The workflow converts MDX to Docusaurus format and deploys to GitHub Pages
3. Enable GitHub Pages in your repository settings (Settings > Pages > Source: GitHub Actions)

### Option 3: Self-Hosted

Run locally for development:

```bash
# Install Mintlify CLI
npm install -g mintlify

# Start local server
cd docs
mintlify dev
```

## Documentation Structure

```
docs/
├── introduction.mdx       # Landing page
├── quickstart.mdx         # Getting started guide
├── supported-models.mdx   # LLM provider documentation
├── customize/             # Customization guides
│   ├── agent/            # Agent configuration
│   ├── browser/          # Browser settings
│   ├── tools/            # Custom tools
│   └── ...
├── examples/             # Example code and apps
└── development/          # Contributing guides
```

## Contributing to Docs

1. Edit MDX files directly
2. Test locally with `mintlify dev`
3. Submit a pull request

## Links

- **Repository**: https://github.com/billy-enrizky/openbrowser-ai
- **Documentation**: https://openbrowser.github.io/docs (GitHub Pages)
- **Issues**: https://github.com/billy-enrizky/openbrowser-ai/issues

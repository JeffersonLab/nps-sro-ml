# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

## docs 

This is a documentation website for the NPS (Neutral Particle Spectrometer) Streaming Readout Machine Learning project. The project applies Graph Neural Network (GNN) methods to streaming-like data for physics analysis, focusing on clustering and classification of signal over background in the NPS detector.

VitePress-based documentation site with markdown content, tutorials, and resources for the NPS SRO ML project at Jefferson Lab.

### Documentation structure

```
nps-sro-ml/
├── docs/                    # VitePress documentation site
│   ├── .vitepress/         # VitePress configuration
│   │   ├── config.mts      # Main site configuration (sidebar, nav, theme)
│   │   └── theme/          # Custom theme components
│   ├── tutorials/          # Analysis tutorials and code examples
│   ├── public/             # Static assets (images, PDFs, etc.)
│   ├── index.md           # Homepage
│   ├── data.md            # Data access documentation
│   ├── resources.md       # Links to VTP manual and resources
│   └── manage-website.md  # Website management guide
├── .github/
│   └── workflows/
│       └── documentation.yml  # GitHub Actions for auto-deployment
└── README.md              # Points to full documentation
```

## Common Commands

All commands must be run from the `docs/` directory:

```bash
cd docs
```

### Install Dependencies
```bash
npm install
```

### Development Server
```bash
npm run dev
```
This starts a local development server (typically at localhost:5173) with hot reload.

### Build for Production
```bash
npm run build
```
Builds the site to `docs/.vitepress/dist/`

### Preview Production Build
```bash
npm run serve
```

## Documentation Architecture

### VitePress Configuration

The main configuration is in `docs/.vitepress/config.mts`:
- Site title, description, and base URL
- Navigation menu and sidebar structure
- Theme customization (logo, footer, social links)
- Markdown plugins: KaTeX (math rendering), Mermaid (diagrams), Lightbox (images)
- Search configuration (local search enabled)

### Content Organization

**Markdown Files:** Located in `docs/` directory
- Use frontmatter for page-specific configuration
- Support KaTeX for mathematical expressions
- Support Mermaid diagrams
- Images can be zoomed with lightbox plugin

**Static Assets:** Located in `docs/public/`
- Files are copied to site root during build
- Reference in markdown without `docs/public/` prefix
- Example: `![image](/path/to/image.png)` for file at `docs/public/path/to/image.png`

**Sidebar Structure:**
Defined in `docs/.vitepress/config.mts` under `themeConfig.sidebar`. When adding new pages:
1. Create markdown file in `docs/`
2. Add entry to sidebar configuration
3. Use appropriate nesting under relevant sections

### Naming Conventions

**Files:** Use lowercase with hyphens or underscores
- `data.md`, `manage-website.md`
- Tutorial files may use underscores: `py-edm4eic-uproot.md`

**Images:** Prefix with numbers for ordering in analysis results
- `01_distribution.png`, `02_spectrum.png`

### Analysis Results Structure

When documenting analysis results:
- Place markdown in: `docs/campaign-YYYY-MM/<analysis-name>.md`
- Place plots in: `docs/public/analysis/campaign-YYYY-MM/<topic>/<energy>/`
- Reference images with absolute paths from `public/`: `![desc](/analysis/campaign-YYYY-MM/topic/energy/image.png)`

## Deployment

The site auto-deploys to GitHub Pages on every push to `main` branch via GitHub Actions (`.github/workflows/documentation.yml`):
1. Workflow triggers on push to main
2. Installs Node.js dependencies
3. Builds VitePress site
4. Uploads to GitHub Pages
5. Deploys to https://jeffersonlab.github.io/nps-sro-ml/

Check deployment status at: https://github.com/JeffersonLab/nps-sro-ml/actions

## Key Technologies

- **VitePress**: Static site generator (v1.6.3)
- **Node.js**: Required version 20
- **Markdown-it**: Enhanced markdown with plugins
- **KaTeX**: Math rendering (v0.16.22)
- **Mermaid**: Diagram generation (v11.6.0)
- **Lightbox**: Image zoom functionality

## Project Context

The NPS SRO ML project focuses on:
- Graph Neural Network (GNN) architectures for calorimeter data analysis
- Using VTP (VETROC Trigger Processor) trigger information as ground truth
- Edge classification using waveform data
- GravNet and transformer-based ConvNet architectures

Data is hosted on JLab ifarm at `/cache/hallc/c-nps/analysis/pass2/replays/updated`

Key branches from ROOT files:
- `NPS.cal.fly.adcSampWaveform` - Waveforms from fADC250
- `NPS.cal.vtpClusX, Y, Time` - VTP cluster positions
- `NPS.cal.fly.block_clusterID` - Clusters found by hcana

## Working with Documentation

When editing documentation:
1. All content changes should be in markdown files
2. Preview locally with `npm run dev` before committing
3. Images go in `docs/public/` and are referenced without that prefix
4. The sidebar is manually configured - update `config.mts` when adding new pages
5. Math expressions use KaTeX syntax: `$inline$` or `$$block$$`
6. Mermaid diagrams use fenced code blocks with `mermaid` language tag
7. The site automatically rebuilds on push to main - check Actions tab for build errors

## Important Configuration Details

### Dead Link Checking

VitePress checks for broken links by default. The configuration includes `ignoreDeadLinks` patterns to allow links to static assets like PDFs and PowerPoint files in `docs/public/`:

```typescript
ignoreDeadLinks: [
    /\.(pdf|pptx|docx|xlsx|zip|tar|gz)$/i
]
```

When adding new types of downloadable files, add their extensions to this pattern.

### WSL/Cross-Platform Issues

If you encounter rollup module errors like `Cannot find module @rollup/rollup-linux-x64-gnu`, this is due to npm's handling of optional dependencies in cross-platform environments (especially WSL). Fix by:

```bash
cd docs
rm -rf node_modules package-lock.json
npm install
```

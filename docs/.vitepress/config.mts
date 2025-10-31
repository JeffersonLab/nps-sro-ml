import markdownItKatex from 'markdown-it-katex';
import { withMermaid } from "vitepress-plugin-mermaid";
import lightbox from "vitepress-plugin-lightbox"


export default withMermaid({
    title: 'NPS-SRO-ML',
    description: 'Documentation for NPS SRO ML analysis',
    base: '/nps-sro-ml/',

    // Ignore dead links for static assets (PDFs, PowerPoint, etc.)
    ignoreDeadLinks: [
        /\.(pdf|pptx|docx|xlsx|zip|tar|gz)$/i
    ],

    // Improved head settings with proper KaTeX styling
    head: [
        ['link', { rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.css' }],
        ['link', { rel: 'icon', type: 'image/png', href: '/favicon.png' }],
        ['meta', { name: 'viewport', content: 'width=device-width, initial-scale=1.0' }],
    ],

    // Enhanced theme configuration
    themeConfig: {
        // Logo (create a simple logo and place it in docs/public/)
        logo: 'logo.png',

        // Improved navigation
        nav: [
            { text: 'Home', link: '/' },
            {
                text: 'Resources',
                items: [
                    { text: 'Meetings', link: 'https://wiki.jlab.org/cuawiki/index.php/EIC_Meson_SF_Meeting_Material_and_Summaries' },
                    { text: 'Data', link: '/data' },
                    { text: 'GitHub', link: 'https://github.com/JeffersonLab/nps-sro-ml' }
                ]
            },
        ],

        // Expanded sidebar with better organization
        sidebar: [
            {
                text: 'Getting Started',
                collapsed: false, // Ensure this is not collapsed
                items: [
                    { text: 'About', link: '/' },
                ]
            },
            {
                text: 'Data',
                link: '/data',
                items: [
                    { text: 'Data Access', link: '/data' },
                ]
            },
            {
                text: 'Other',
                items: [
                    { text: 'Resources', link: '/resources' },
                    { text: 'Manage website', link: '/manage-website' },
                ]
            }
        ],

        // Footer customization
        footer: {
            message: 'Released under the MIT License.',
            copyright: 'Copyright © 2025 Meson Structure Collaboration'
        },

        // Social links
        socialLinks: [
            { icon: 'github', link: 'https://github.com/JeffersonLab/sro-nps-ml' }
        ],

        // Search configuration
        search: {
            provider: 'local'
        },

        // Layout customization for large screens
        outline: {
            level: [2, 3],
            label: 'On this page'
        },

        // Additional helpful features
        editLink: {
            pattern: 'https://github.com/JeffersonLab/nps-sro-ml/edit/main/docs/:path',
            text: 'Edit this page on GitHub'
        },

        // Dark/Light theme toggle (enabled by default)
        appearance: true
    },

    // Enable KaTeX for math rendering
    markdown: {
        config: (md) => {
            md.use(markdownItKatex);
            md.use(lightbox, {});
        }
    },

    // Fix layout issues on large screens
    vite: {
        css: {
            preprocessorOptions: {
                scss: {
                    additionalData: `
            // Add any global SCSS variables here
          `
                }
            }
        }
    }
});
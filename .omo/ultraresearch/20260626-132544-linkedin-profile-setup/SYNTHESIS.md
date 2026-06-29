# Ultraresearch Synthesis: LinkedIn Profile Setup

Workers: 6 attempted · Waves: 2 · Sources: LinkedIn Help, LinkedIn Talent/NACE/career guidance, GitHub, Hugging Face API/pages, W&B URL check, local repos · Verifications: browser UI inspection, public URL checks, local evidence map

## Executive Summary

The current LinkedIn profile is under-positioned for job search because its headline is only `동아대학교 학생`, the About section appears empty, and LinkedIn itself is prompting for industry and summary completion. The profile already has Open to Work enabled, which gives a base recruiting signal, but it needs recruiter-searchable keywords and proof links.

The best profile direction is a skills-indexed proof page: top-line keywords for AI/ML, software engineering, LLM/MLOps, agentic AI, and GovTech; a grounded About section naming public projects; Featured links to proof surfaces; and a defensible skills set. This follows LinkedIn's own distinction between headline/intro/About/Featured/Skills/Open-to-Work and 2026 recruiter guidance toward skills-based hiring.

## Profile Fields To Update

- Headline: `AI/ML & Software Engineering Student | K-EXAONE Apps · Tool-Calling Agents · Public-Service AI | UMMAYA · GovOn HF · DigitalPublishing`
- Industry: `소프트웨어 개발` or LinkedIn's closest `Software Development` option.
- About: use the text in `PROFILE_UPDATE_PACKAGE.md`.
- Featured links: GitHub profile, UMMAYA docs, Hugging Face, DigitalPublishing live demo, edu archive, W&B GovOn.
- Skills: Python, JavaScript, TypeScript, Software Engineering, Machine Learning, Deep Learning, LLM, AI Agents, Hugging Face, Hugging Face Transformers, PyTorch, Model Fine-Tuning, LoRA, QLoRA, Quantization, RAG, vLLM, SQL, GitHub, Docker, Three.js, WebGL, HTML, CSS, Data Structures, Algorithms, Operating Systems, Computer Vision.

## Evidence By Theme

### LinkedIn Mechanics

- LinkedIn official docs confirm headline editing from the intro card and separate About editing. Sources: https://www.linkedin.com/help/linkedin/answer/a542926/edit-your-headline?lang=en, https://www.linkedin.com/help/linkedin/answer/a553140.
- LinkedIn recommends the About section express mission, motivation, and skills, ideally in one or two paragraphs. Source: https://www.linkedin.com/help/linkedin/answer/a554351.
- Featured supports work samples, authored posts, external links, documents, media, and videos, but is not search-discoverable. Sources: https://www.linkedin.com/help/linkedin/answer/a552452/featured-section-on-your-profile-faqs, https://www.linkedin.com/help/linkedin/answer/a550399/manage-featured-samples-of-your-work-on-your-profile.
- Skills can be added and reordered, with LinkedIn supporting up to 100 skills. Sources: https://www.linkedin.com/help/linkedin/answer/a549047/add-and-remove-skills-on-your-profile, https://www.linkedin.com/help/linkedin/answer/a568137.
- Open to Work supports public/all-member visibility or recruiter-only visibility, with privacy caveats. Sources: https://www.linkedin.com/help/linkedin/answer/a507508/let-recruiters-know-you-re-open-to-work?lang=en, https://www.linkedin.com/help/linkedin/answer/a510407.

### 2026 Recruiting Pattern

- The profile should not leave the headline as only student status; the headline is a recruiter-facing search and positioning surface.
- 2026 hiring guidance emphasizes skills-based screening, so explicit skills should appear in headline/About/Skills, backed by proof links.
- Featured should support human review after the click, not replace keyword fields.
- Sources: https://www.linkedin.com/help/linkedin/answer/a793433, https://www.linkedin.com/business/talent/blog/talent-acquisition/business-case-for-skills-first-hiring, https://www.naceweb.org/job-market/trends-and-predictions/employer-use-of-skills-based-hiring-practices-grows, https://www.naceweb.org/job-market/trends-and-predictions/demand-for-ai-skills-in-entry-level-jobs-nearly-triples-since-fall-2025, https://studentsuccess.utk.edu/career/students/networking/your-linkedin-profile/.

### Public Project Evidence

- UMMAYA is public and describes a terminal AI agent for Korean public-service workflows with tool progress and identity/consent/payment/authority boundaries. Evidence: `/Users/um-yunsang/UMMAYA/README.md:12-24`, `/Users/um-yunsang/UMMAYA/README.md:83-115`; public repo: https://github.com/umyunsang/UMMAYA.
- UMMAYA docs are public and accessible. Verified URL: https://ummaya-docs.pages.dev/en/.
- Hugging Face profile has public GovOn/EXAONE-related models, datasets, and Spaces. Verified via public page/API: https://huggingface.co/umyunsang, https://huggingface.co/umyunsang/GovOn-EXAONE-LoRA-v2, https://huggingface.co/umyunsang/GovOn-EXAONE-AWQ-v2, https://huggingface.co/datasets/umyunsang/govon-civil-response-data.
- W&B public GovOn URL responds, but detailed metadata was not extracted without rendered client state. Use link-only conservative wording. Verified URL: https://wandb.ai/umyun3/GovOn.
- DigitalPublishing is a public class/portfolio workspace and includes a Three.js/WebGL mobile invitation live demo. Evidence: `/Users/um-yunsang/Documents/DigitalPublishing/README.md:7-21`, `/Users/um-yunsang/Documents/DigitalPublishing/README.md:29-33`, `/Users/um-yunsang/Documents/DigitalPublishing/mobile-wedding-unrolling-invitation/README.md:3-16`, `/Users/um-yunsang/Documents/DigitalPublishing/mobile-wedding-unrolling-invitation/README.md:34-57`; public demo: https://ourseason.pages.dev/.
- edu is a public CS/AI curriculum archive with graph interfaces and domain folders. Evidence: `/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu/README.md:30-41`, `/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu/README.md:147-158`; public repo: https://github.com/umyunsang/edu.
- LG Aimers 8th-cycle notes support LLM Compression and EXAONE/lightweight-model learning. Evidence: `/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu/LGAimer/LG Aimers 8기/LG Aimers 8기.md:36-57`.
- IlluOps must be framed as planning/evidence/reference, not a shipped app. Evidence: `/Users/um-yunsang/IlluOps/AGENTS.md:7-19`, `/Users/um-yunsang/IlluOps/AGENTS.md:44-52`; public repo: https://github.com/umyunsang/IlluOps.

## Browser UI Findings

- Current profile URL: `linkedin.com/in/윤상-엄-b54725419/?skipRedirect=true`.
- Intro edit modal exposes editable first name, last name, headline, country/region, education, and industry fields.
- Current headline field contains `동아대학교 학생`.
- The profile page prompted for industry and About/summary completion.
- No changes were typed or saved during inspection.

## Gaps And Boundaries

- Do not claim exact LinkedIn recruiter ranking weights; public docs confirm fields and filters but not scoring.
- Do not claim IlluOps is a finished product.
- Do not claim W&B metrics; only the public URL is verified.
- Do not claim official government affiliation for UMMAYA.
- Do not publish private `.omo` evidence as public proof.

## Expansion Trace

- Wave 1 covered LinkedIn official mechanics, 2026 recruiter strategy, local evidence, and public portfolio surfaces.
- Wave 2 opened role-keyword taxonomy and adversarial claim review.
- Adversarial review rejected the initial `MLOps/GovTech/W&B` phrasing as too broad. The final package uses `K-EXAONE Apps`, `Tool-Calling Agents`, `Public-Service AI`, and `GovOn HF` instead.
- The profile package is ready for action-time user confirmation before any public LinkedIn typing/saving.

# SatQuery AI design system

This frontend uses a restrained academic workspace: navy headings, blue actions, white surfaces and thin borders. The source of truth for reusable values is `frontend/src/styles/tokens.css`; primitive styles live in `frontend/src/styles/ui.css`, and application layouts in `frontend/src/styles.css`.

## Non-negotiable honesty rules

Never remove or weaken honesty labels while changing the layout. Every result retains its **Real**, **Stub · synthetic output**, or **Pre-computed · not live inference** label. Pre-computed placeholders must remain explicitly marked as placeholders. Preserve validation status, Grounding's **unvalidated** notice, **uncalibrated** scores, **not available** confidence/evidence, Model 4's **not verified on a real pair** caveat, and domain-gap warnings. Do not turn heuristic routing scores into accuracy claims. Availability of weights is not proof of quality.

The global caveat notice may collapse and remembers its state in local storage (`satquery.ui.notice-expanded`). Its collapsed summary still names the unvalidated grounding model and unverified fusion model. Collapsing it must never hide result-specific labels. Results keep their warnings, input requirements and execution traces.

Read modes and validation text from the existing API. Do not hardcode a model's mode from an old demo or slide. The device cluster shows configured selection (`auto` when unknown), or the device of the last live result, explicitly marked `last`. The API does not expose active-loaded-device telemetry; never imply that a device is currently running a loaded model.

## Tokens

| Purpose | Value |
| --- | --- |
| Background / surface / border | `#F4F6F9` / `#FFFFFF` / `#D8DFE8` |
| Text / muted text | `#0F1B2D` / `#5A6B7F` |
| Primary / hover / accent | `#1F4E79` / `#173C5E` / `#0070C0` |
| Success / tint / text | `#2E7D5B` / `#EAF5EF` / `#236447` |
| Warning / tint / text | `#B26A00` / `#FFF4DF` / `#7A4900` |
| Danger / tint | `#B3261E` / `#FDEEEB` |
| Info / tint | `#2A6FA8` / `#EAF3FB` |
| Control border / selected background | `#748398` / `#EAF1F8` |

Spacing tokens are 8, 16, 24, 32, 40, 48 and 64 px. Use the 8 px grid for layout, padding and gaps. Standard radius is 8 px; chips and small labels use 6 px. Use 1 px borders, not shadows. The lighter border is for surface separation; controls have the darker border to remain identifiable. Focus uses a 2 px accent outline with offset; never suppress it without an equally visible replacement.

The warning accent on white is only 4.24:1, so it is not small-text ink. Use the darker warning text on the warning tint. Likewise, success text uses the darker success ink. Status is always conveyed with words, not colour alone.

### Text contrast

Ratios below are calculated from the WCAG relative-luminance formula, rounded to two decimals. Normal text must reach 4.5:1, including micro-labels. These checks cover the design's defined text/background pairs; imagery and backend-rendered evidence annotations are not changed by this frontend redesign.

| Text / background | Contrast |
| --- | ---: |
| Text / white | 17.28:1 |
| Muted / page background | 5.05:1 |
| White / primary | 8.66:1 |
| Warning text / warning tint | 6.93:1 |
| Success text / success tint | 6.30:1 |
| Danger / danger tint | 5.79:1 |
| Info / info tint | 4.75:1 |
| Accent link / white | 5.15:1 |

Also retain readable text on disabled controls; do not fade whole controls with opacity. See [WCAG contrast minimum](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html).

## Typography and icons

Fonts are self-hosted through `@fontsource/ibm-plex-sans` and `@fontsource/ibm-plex-mono`; no font CDN or remote stylesheet. Sans weights are 400, 500 and 600, Mono weights 400 and 500. UI base is 15 px with 1.55 line height. Headings are semibold (600), with a 32 / 20 / 18 px scale. Supporting text is 13 px. Micro-labels are 11–12 px, uppercase, with 0.08 em tracking.

Use Plex Mono for request IDs, versions, coordinates, JSON and traces. Metrics use tabular numerals. Do not make long IDs or trace JSON force the page wider; allow wrapping or a dedicated scroll region.

Icons come from `lucide-react`, 16–18 px with stroke width 1.75. Decorative icons have `aria-hidden`; icon-only buttons require an accessible name. No emoji as UI icons. The navy SVG mark and favicon share the same simple layered-map geometry.

## Shared components

Import primitives from `frontend/src/components/ui`.

| Component | API and behaviour |
| --- | --- |
| `Button` | `variant="primary|secondary|ghost"`, `loading`, `disabled`, optional `icon`, native button props and forwarded ref. Default type is `button`; explicitly use `submit` for forms. Loading disables repeat activation and sets `aria-busy`; keep descriptive loading text. Icons remain static. |
| `Badge` / `Chip` | `tone="neutral|success|warning|danger|info"`. Badge is non-interactive text. Chip becomes a native button when `onClick` is supplied, supports `selected` / `aria-pressed` and `disabled`. Never make a passive badge a keyboard stop. |
| `Card` | `as` selects the semantic element (default `section`); supports native props/ref. White surface, border and padding only. Give sections a meaningful heading. |
| `SegmentedControl` | `label`, `options`, `value`, `onChange`, `disabled`; options have value, label and optional disabled. Native radios give Tab and arrow-key support; visible focus surrounds the selected control label. |
| `Tabs` | `label`, `items`, `value`, `onChange`; each item has value, label, content and optional disabled. Use valid enabled selected values. Tab enters the active tab, arrows/Home/End move and activate, disabled tabs are skipped. Panels are labelled and focusable. |
| `Tooltip` | `content` and one focusable child that forwards event/ARIA props. Appears on hover or focus; Escape dismisses. Essential instructions must also exist as ordinary text. |
| `Drawer` | `open`, `onClose`, `title`, children and optional footer. Native modal dialog makes the background inert, contains keyboard focus, focuses its heading, closes with Escape/close button/backdrop, and restores opener focus. Body scroll is locked while open. |
| `Skeleton` | `label`, `lines`, optional class. Static placeholders with a screen-reader status label, no pulsing animation. |
| `Toast` | `tone`, `title`, children and optional `onDismiss`. Danger uses alert semantics; other messages use status. Optional named close button and Escape dismissal. Do not auto-dismiss information users need to read. |
| `EmptyState` | `title`, `description`, optional icon, children and class. Present a clear next action where useful; never imply evidence or confidence is guaranteed. |

Native controls provide Enter/Space activation. All interactive controls have visible keyboard focus. Informational containers, badges and skeleton shapes are not unnecessary tab stops. Modal behaviour follows the [WAI modal-dialog pattern](https://www.w3.org/WAI/ARIA/apg/patterns/dialog-modal/).

## Shell and layouts

The slim header contains the mark, Workspace/History links, backend connection label, truthful device status and Models drawer trigger. Existing hash routes remain `#workspace` and `#history`; `#limitations` adds a frontend static limitations view. Navigation uses `aria-current`. The Models drawer reuses the existing registry response; configuration controls and expanded management workflows are outside this redesign.

The workspace separates input controls and analysis output. At smaller widths it becomes one column; upload slots and the header wrap without hiding their meaning. History retains pagination and opens the existing result view. Trace JSON remains selectable, scrollable and downloadable. Downloads, upload inspection, overrides, polling, cancellation and report behaviour continue to use the existing API.

The footer is one slim line: “SatQuery AI - Academic demonstration - SIH26167”, plus Limitations. The limitations view is a static frontend copy of project caveats; keep it aligned with `docs/limitations.md` as facts change.

The faint topographic SVG is permitted **only** in the header background and empty states. It is decorative, low contrast and conveys no information. No gradients, glow, glassmorphism, blur, heavy shadows, decorative animation or animated skeletons/spinners. Use whitespace, typography, alignment and borders for hierarchy.

## Verification for this redesign

Existing frontend tests and the Vite production build must pass. Review desktop and narrow layouts, keyboard focus and Escape in the drawer, persistent notice collapse, history, limitations, and result honesty labels. Backend, registry, API contracts and existing tests are outside the redesign scope.

import json
from pydantic import BaseModel, Field
from playwright.async_api import Page
from langchain_core.language_models.chat_models import BaseChatModel
from url_finder import find_candidate_urls


class JobLink(BaseModel):
    url: str = Field(description="The full URL of the job vacancy.")
    text: str = Field(description="The visible anchor text or title of the vacancy link.")


class BrowserTools:
    def __init__(self, page: Page):
        self.page = page
        self.state_version = 0
        self.state_valid = False
        # element_id -> Frame that contains it (main frame or an iframe).
        self._element_frames = {}

    async def get_page_state(self) -> dict:
        """Retrieves the current page state and assigns versioned element IDs.

        Increments self.state_version and evaluates page state in the browser
        context, annotating visible interactive elements with an
        'element-id' DOM attribute.

        Returns:
            dict: A dictionary with the following structure:

            {
                "state_version": 1,
                "url": "https://example.com/page",
                "title": "Page Title",
                "elements": [
                    {
                        "element_id": 1,
                        "type": "button",
                        "name": "Submit",
                        "disabled": false
                    }
                ]
            }

            The `element_id`s are only valid for the returned state_version.

            Type-specific properties:
            - href: present for links.
            - value, placeholder: present for text inputs.
            - checked: present for checkboxes and radio buttons.
            - options: present for dropdowns; contains at most 30 options.
            - frame: present for elements inside an iframe (the frame URL).

            Elements inside open shadow roots and visible iframes are
            included. Element types: button, dropdown, checkbox, radio,
            text_input, link, clickable (ARIA options/tabs/menu items).
        """

        self.state_version += 1
        self.state_valid = True

        print(
            f"\n[*] Retrieving page state "
            f"(version {self.state_version})"
        )

        js_script = """
        (startId) => {

            // Collect the document and every open shadow root so elements
            // inside web components can be found.
            const roots = [document];

            for (let i = 0; i < roots.length; i++) {
                roots[i].querySelectorAll('*').forEach(el => {
                    if (el.shadowRoot) roots.push(el.shadowRoot);
                });
            }

            const queryAll = (selector) =>
                roots.flatMap(root =>
                    Array.from(root.querySelectorAll(selector))
                );

            const isVisible = (el) => {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);

                return (
                    rect.width > 0 &&
                    rect.height > 0 &&
                    style.display !== 'none' &&
                    style.visibility !== 'hidden' &&
                    parseFloat(style.opacity) > 0 &&
                    style.pointerEvents !== 'none'
                );
            };

            const isBlockedInput = (el) => {
                const text = [
                    el.placeholder,
                    el.value,
                    el.name,
                    el.id,
                    el.getAttribute('aria-label'),
                    el.getAttribute('title')
                ]
                .filter(Boolean)
                .join(' ')
                .toLowerCase();

                return [
                    'cookie',
                    'privacy',
                    'newsletter',
                    'subscribe',
                    'consent'
                ].some(term => text.includes(term));
            };

            const getName = (el) => {
                // Explicit label
                if (el.id) {
                    const label = el.getRootNode().querySelector(
                        `label[for="${CSS.escape(el.id)}"]`
                    );

                    if (label) {
                        const text = label.innerText
                            .replace(/\\s+/g, ' ')
                            .trim();

                        if (text) return text;
                    }
                }

                // Label wrapping the element
                const parentLabel = el.closest('label');

                if (parentLabel) {
                    const text = parentLabel.innerText
                        .replace(/\\s+/g, ' ')
                        .trim();

                    if (text) return text;
                }

                // Accessible name
                const ariaLabel = el.getAttribute('aria-label');

                if (ariaLabel) {
                    return ariaLabel.trim();
                }

                // Title
                const title = el.getAttribute('title');

                if (title) {
                    return title.trim();
                }

                // Placeholder for text inputs
                if (el.placeholder) {
                    return el.placeholder.trim();
                }

                // Visible text
                const text = el.innerText
                    .replace(/\\s+/g, ' ')
                    .trim();

                if (text) {
                    return text.substring(0, 120);
                }

                // Input value as final fallback
                if (el.value) {
                    return String(el.value)
                        .replace(/\\s+/g, ' ')
                        .trim()
                        .substring(0, 120);
                }

                return '';
            };

            // Remove IDs left over from previous states so that
            // [element-id='N'] always resolves to exactly one element.
            queryAll('[element-id], [element-proxy]').forEach(el => {
                el.removeAttribute('element-id');
                el.removeAttribute('element-proxy');
            });

            const interactive = [];
            const tagged = new Set();

            let elementId = startId;

            const addElement = (el, type, extra = {}, proxy = false) => {
                // An element can match several groups (e.g. <a role="button">);
                // expose it only once.
                if (tagged.has(el)) return;
                if (!proxy && !isVisible(el)) return;

                tagged.add(el);

                // Visually hidden control operated through a visible label.
                if (proxy) el.setAttribute('element-proxy', '1');

                el.setAttribute('element-id', elementId);

                interactive.push({
                    element_id: elementId,
                    type: type,
                    name: getName(el),
                    disabled: !!el.disabled,
                    ...extra
                });

                elementId++;
            };

            /*
            * 1. Buttons
            *
            * Includes native buttons and elements using role="button".
            * Cookie/consent buttons remain visible.
            */
            queryAll(
                'button, [role="button"]'
            ).forEach(el => {
                addElement(el, 'button');
            });

            /*
            * 2. Native dropdowns
            *
            * Only visible <select> elements are exposed.
            * Hidden/transparent backing selects for custom dropdowns
            * are ignored.
            */
            queryAll('select').forEach(el => {
                if (!isVisible(el)) return;

                addElement(el, 'dropdown', {
                    options: Array.from(el.options)
                        .slice(0, 30)
                        .map(option => ({
                            name: option.text.trim(),
                            value: option.value,
                            selected: option.selected
                        }))
                });
            });

            /*
            * Custom-styled checkboxes/radios often hide the real <input>
            * (opacity 0 / 0px) and show a label instead. Expose such an
            * input when its label is visible.
            */
            const addToggle = (el, type) => {
                const extra = { checked: !!el.checked };

                if (el.tagName === 'INPUT' && !isVisible(el)) {
                    const label =
                        (el.id && el.getRootNode().querySelector(
                            `label[for="${CSS.escape(el.id)}"]`
                        )) || el.closest('label');

                    if (label && isVisible(label)) {
                        addElement(el, type, extra, true);
                        return;
                    }
                }

                addElement(el, type, extra);
            };

            /*
            * 3. Checkboxes
            */
            queryAll(
                'input[type="checkbox"], [role="checkbox"]'
            ).forEach(el => {
                if (
                    el.tagName === 'INPUT' &&
                    isBlockedInput(el)
                ) {
                    return;
                }

                addToggle(el, 'checkbox');
            });

            /*
            * 4. Radio buttons
            */
            queryAll(
                'input[type="radio"], [role="radio"]'
            ).forEach(el => {
                addToggle(el, 'radio');
            });

            /*
            * 5. Text inputs
            *
            * Only inputs intended for textual entry are exposed.
            */
            queryAll(
                'input, textarea'
            ).forEach(el => {

                const tag = el.tagName.toLowerCase();
                const inputType = (el.type || 'text').toLowerCase();

                const isTextInput =
                    tag === 'textarea' ||
                    [
                        'text',
                        'search',
                        'email',
                        'tel',
                        'url'
                    ].includes(inputType);

                if (!isTextInput) return;

                // Do not expose cookie/privacy/newsletter inputs.
                if (tag === 'input' && isBlockedInput(el)) {
                    return;
                }

                addElement(el, 'text_input', {
                    value: el.value || '',
                    placeholder: el.placeholder || ''
                });
            });

            /*
            * 6. Links
            */
            queryAll('a').forEach(el => {
                if (!el.href) return;

                addElement(el, 'link', {
                    href: el.href
                });
            });

            /*
            * 7. Other clickable widgets
            *
            * Custom dropdown options, tabs, menu items, etc. that are not
            * native buttons or links.
            */
            queryAll(
                '[role="option"], [role="menuitem"], [role="tab"], ' +
                '[role="combobox"], [role="switch"], [role="link"], summary'
            ).forEach(el => {
                addElement(el, 'clickable');
            });

            return {
                url: window.location.href,
                title: document.title,
                elements: interactive
            };
        }
        """

        main_frame = self.page.main_frame
        frames = [main_frame] + [
            f for f in self.page.frames if f is not main_frame
        ]

        self._element_frames = {}
        elements = []
        main_state = None
        next_id = 1

        for frame in frames:
            if frame is not main_frame and not await self._frame_is_visible(
                frame
            ):
                continue

            try:
                state = await frame.evaluate(js_script, next_id)
            except Exception as e:
                if frame is main_frame:
                    raise

                # Frame navigated or detached mid-evaluation.
                print(f"[!] Skipping frame {frame.url}: {e}")
                continue

            if frame is main_frame:
                main_state = state

            for element in state["elements"]:
                if frame is not main_frame:
                    element["frame"] = frame.url

                self._element_frames[element["element_id"]] = frame
                elements.append(element)

            next_id += len(state["elements"])

        result = {
            "state_version": self.state_version,
            "url": main_state["url"],
            "title": main_state["title"],
            "elements": elements,
        }

        return result

    @staticmethod
    async def _frame_is_visible(frame) -> bool:
        """Whether a child frame's <iframe> element is rendered."""
        try:
            handle = await frame.frame_element()
            return await handle.is_visible()
        except Exception:
            return False

    async def interact(
        self,
        state_version: int,
        element_id: int,
        action: str,
        value: str = ""
    ) -> dict:
        """
        Interacts with an element from the current page state.

        element_id is only valid for the specified state_version.

        Args:
            state_version: Page state version associated with element_id.
            element_id: ID of the element to interact with.
            action: Action to perform: click, fill, check, or select.
            value: Value used by fill or select.

        Returns:
            dict: A dictionary with the following structure:

                {
                    "success": true,
                    "element_id": 1,
                    "action": "click",
                    "navigation": {
                        "from": "https://example.com",
                        "to": "https://example.com/next"
                    },
                    "error": null,
                    "message": null,
                    "next_step": "get_page_state"
                }

            On failure, success is false and error/message describe the
            failure. Fields not applicable to the outcome are null.

            Possible error codes include STALE_STATE, ELEMENT_NOT_FOUND,
            NOT_NATIVE_SELECT, INVALID_ACTION, and INTERACTION_FAILED.

            next_step instructs the agent which tool to call next and is
            always get_page_state.
        """

        # Reject stale state.
        if state_version != self.state_version or not self.state_valid:
            return {
                "success": False,
                "element_id": element_id,
                "action": action,
                "navigation": None,
                "error": "STALE_STATE",
                "message": (
                    f"State {state_version} is stale. "
                    f"Current state is {self.state_version}."
                ),
                "next_step": "get_page_state"
            }

        action = (action or "").strip().lower()

        try:
            # Elements may live in an iframe; Playwright CSS locators
            # already pierce open shadow roots.
            frame = self._element_frames.get(
                element_id, self.page.main_frame
            )

            locator = frame.locator(
                f"[element-id='{element_id}']"
            ).first

            # Check existence first: evaluate() on a missing element would
            # block for the full default timeout (30s) before failing.
            visible = (
                await locator.count() > 0
                and await locator.is_visible()
            )

            # Visually hidden checkbox/radio inputs driven by a visible
            # label are tagged 'element-proxy' by get_page_state.
            is_proxy = (
                not visible
                and not frame.is_detached()
                and await locator.count() > 0
                and await locator.get_attribute("element-proxy") is not None
            )

            if frame.is_detached() or not (visible or is_proxy):
                return {
                    "success": False,
                    "element_id": element_id,
                    "action": action,
                    "navigation": None,
                    "error": "ELEMENT_NOT_FOUND",
                    "message": (
                        f"Element ID {element_id} was not found."
                    ),
                    "next_step": "get_page_state"
                }

            target = await locator.evaluate("""
            (el) => {
                const label = el.id
                    ? el.getRootNode().querySelector(
                        `label[for="${CSS.escape(el.id)}"]`
                    )
                    : null;

                return (
                    label?.innerText ||
                    el.closest('label')?.innerText ||
                    el.getAttribute('aria-label') ||
                    el.getAttribute('title') ||
                    el.placeholder ||
                    el.innerText ||
                    el.value ||
                    ''
                )
                .replace(/\\s+/g, ' ')
                .trim()
                .substring(0, 120);
            }
            """)

            print(
                f"\n[*] Interacting "
                f"| state={state_version}:{element_id} "
                f"| action={action.upper()} "
                f"| target='{target}' "
                f"| value='{value}'"
            )

            prev_url = self.page.url

            if action == "click":
                await locator.click(timeout=5000, force=is_proxy)

            elif action == "fill":
                await locator.fill(value, timeout=5000)

            elif action == "check":
                await locator.check(timeout=5000, force=is_proxy)

            elif action == "select":
                if await locator.evaluate(
                    "(el) => el.tagName.toLowerCase()"
                ) != "select":
                    return {
                        "success": False,
                        "element_id": element_id,
                        "action": action,
                        "navigation": None,
                        "error": "NOT_NATIVE_SELECT",
                        "message": (
                            "Only native dropdowns support 'select'."
                        ),
                        "next_step": "get_page_state"
                    }

                try:
                    await locator.select_option(
                        label=value,
                        timeout=5000
                    )
                except Exception:
                    # The agent may pass the option's value instead of
                    # its visible label.
                    await locator.select_option(
                        value=value,
                        timeout=5000
                    )

            else:
                return {
                    "success": False,
                    "element_id": element_id,
                    "action": action,
                    "navigation": None,
                    "error": "INVALID_ACTION",
                    "message": (
                        f"Unknown action '{action}'. "
                        "Use click, fill, check, or select."
                    ),
                    "next_step": "get_page_state"
                }

            # The DOM may have changed: any action invalidates the state.
            self.state_valid = False

            # Let navigation / SPA re-rendering settle so the URL read below
            # and the next get_page_state() see the post-action page.
            try:
                await self.page.wait_for_load_state(
                    "domcontentloaded", timeout=5000
                )
                await self.page.wait_for_load_state(
                    "networkidle", timeout=3000
                )
            except Exception:
                # Pages with long-polling/analytics may never go idle.
                pass

            current_url = self.page.url

            return {
                "success": True,
                "element_id": element_id,
                "action": action,
                "navigation": (
                    {
                        "from": prev_url,
                        "to": current_url
                    }
                    if current_url != prev_url
                    else None
                ),
                "error": None,
                "message": None,
                "next_step": "get_page_state"
            }

        except Exception as e:
            print(f"Error in interact: {str(e)} \n")

            # A failed action may still have partially changed the page.
            self.state_valid = False

            return {
                "success": False,
                "element_id": element_id,
                "action": action,
                "navigation": None,
                "error": "INTERACTION_FAILED",
                "message": (
                    f"Failed to perform '{action}' "
                    f"on element_id {element_id}: {str(e)}"
                ),
                "next_step": "get_page_state"
            }

    async def extract_urls(self) -> list[JobLink]:
        """Extract all visible links from the current page with their text and URL.

        Returns:
            list[JobLink]: Vacancy links with their URL and visible text.
        """

        print("\n[*] Extracting links directly from DOM...")

        await self.page.wait_for_timeout(3000)

        js_extract = """
        () => {
            // Include links inside open shadow roots.
            const roots = [document];

            for (let i = 0; i < roots.length; i++) {
                roots[i].querySelectorAll('*').forEach(el => {
                    if (el.shadowRoot) roots[roots.length] = el.shadowRoot;
                });
            }

            const links = roots.flatMap(root =>
                Array.from(root.querySelectorAll('a[href]'))
            );

            return links
                .map(a => ({
                    text: (a.innerText || a.getAttribute('aria-label') || '')
                        .trim()
                        .replace(/\\s+/g, ' '),
                    href: a.href
                }))
                .filter(item =>
                    item.text.length > 0 &&
                    item.href.startsWith('http')
                );
        }
        """

        # Collect from the main frame and every visible iframe
        # (job boards are frequently embedded).
        raw_links = []

        for frame in self.page.frames:
            if (
                frame is not self.page.main_frame
                and not await self._frame_is_visible(frame)
            ):
                continue

            try:
                raw_links.extend(await frame.evaluate(js_extract))
            except Exception as e:
                print(f"[!] Skipping frame {frame.url}: {e}")

        # Deduplicate by URL
        unique_links = {}

        for link in raw_links:
            url = link["href"]
            text = link["text"]

            if url and url not in unique_links:
                unique_links[url] = text

        # Convert to dictionaries
        links = [
            {"url": url, "text": text}
            for url, text in unique_links.items()
        ]

        # Filter using URL patterns and ATS domains
        links = find_candidate_urls(links)

        # Convert to JobLink instances
        job_links = [
            JobLink(url=link["url"], text=link["text"])
            for link in links
        ]

        print(f"\n[*] Extracted {len(job_links)} matching links.")

        return job_links


class MatchingVacancies(BaseModel):
    job_links: list[JobLink] = Field(
        description=(
            "All job vacancies matching the user's criteria. "
            "Include every relevant vacancy, not just the best match."
        )
    )

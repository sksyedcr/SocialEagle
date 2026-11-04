from playwright.sync_api import sync_playwright

# Replace this with the full URL of the specific Cricbuzz match Commentary page.
MATCH_URL = "https://www.cricbuzz.com/live-cricket-scores/161245/nz-vs-ind-1st-odi-india-tour-of-new-zealand-2026"


def get_cricket_score():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page(viewport={"width": 1400, "height": 900})

        try:
            print("Opening Cricbuzz match page...")
            page.goto(MATCH_URL, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(3000)

            # Find the smallest DIV containing both innings score patterns.
            # This avoids accidentally selecting news headlines mentioning India/NZ.
            score_text = page.locator("div").evaluate_all(
                r"""elements => {
                    const hasNZScore = text => /NZ\s+\d+\s*\(/i.test(text);
                    const hasIndiaScore = text => /IND\s+\d+\/\d+\s*\(/i.test(text);

                    const matches = elements
                        .map(el => ({ text: (el.innerText || "").trim() }))
                        .filter(item =>
                            item.text &&
                            hasNZScore(item.text) &&
                            hasIndiaScore(item.text)
                        );

                    matches.sort((a, b) => a.text.length - b.text.length);
                    return matches.length ? matches[0].text : null;
                }"""
            )

            print("\n========== CRICBUZZ SCORE ==========\n")

            if score_text:
                print(score_text)
                page.screenshot(path="score.png", full_page=False)
                print("\nSaved screenshot as score.png")
            else:
                print("Could not find both innings scores.")
                print("Make sure MATCH_URL is the specific match Commentary URL.")
                print("Current page:", page.url)
                page.screenshot(path="debug_page.png", full_page=True)
                print("Saved debug_page.png for troubleshooting.")

            print("\n====================================")

        except Exception as error:
            print("An error occurred:", error)

        finally:
            browser.close()


if __name__ == "__main__":
    if MATCH_URL == "PASTE_THE_MATCH_COMMENTARY_URL_HERE":
        print("Paste the specific Cricbuzz match Commentary URL into MATCH_URL.")
    else:
        get_cricket_score()

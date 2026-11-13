"""Headless integration checks against the existing app's real Streamlit widgets."""
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class DashboardTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file("app.py", default_timeout=60)
        with patch("urllib.request.urlopen", side_effect=OSError("offline")):
            app.run()
            self.assertFalse(app.exception)
            app.sidebar.radio[0].set_value("Recovery Prioritization").run()
        self.assertFalse(app.exception)
        return app

    def test_demo_page_and_weight_change(self):
        app = self.app()
        self.assertEqual(app.dataframe[0].value.iloc[0]["Image ID"], "SYN-B")
        with patch("urllib.request.urlopen", side_effect=OSError("offline")):
            for key, value in [("flood_impact", 100), ("infrastructure_exposure", 1),
                               ("access_disruption", 1), ("rainfall_context", 0)]:
                app.slider(key="recovery_weight_" + key).set_value(value)
            app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.dataframe[0].value.iloc[0]["Image ID"], "SYN-A")

    def test_empty_dataset_and_missing_files(self):
        app = self.app()
        with patch("urllib.request.urlopen", side_effect=OSError("offline")):
            app.radio(key="recovery_mode").set_value("Organizer dataset").run()
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)

    def test_original_ai_helper_still_renders(self):
        app = self.app()
        with patch("urllib.request.urlopen", side_effect=OSError("offline")):
            app.sidebar.radio[0].set_value("AI helper").run()
        self.assertFalse(app.exception)
        self.assertTrue(app.chat_input)

    def test_existing_pages_handle_missing_datasets(self):
        app = self.app()
        with patch("urllib.request.urlopen", side_effect=OSError("offline")):
            for page in ["Explore any data", "311 by neighborhood", "Flood photos"]:
                app.sidebar.radio[0].set_value(page).run()
                self.assertFalse(app.exception, page)

    def test_all_zero_weights_show_error(self):
        app = self.app()
        with patch("urllib.request.urlopen", side_effect=OSError("offline")):
            for key in ("flood_impact", "infrastructure_exposure", "access_disruption", "rainfall_context"):
                app.slider(key="recovery_weight_" + key).set_value(0)
            app.run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)

    def test_human_review_saved(self):
        app = self.app()
        with patch("urllib.request.urlopen", side_effect=OSError("offline")):
            next(item for item in app.selectbox if item.label == "Review status").set_value("Reviewed")
            next(item for item in app.text_input if item.label == "Reviewer").set_value("Test reviewer")
            next(item for item in app.text_area if item.label == "Evidence and verification notes").set_value("Synthetic fixture review only")
            next(item for item in app.button if item.label == "Save review").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.dataframe[0].value.iloc[0]["Human review"], "Reviewed")

import unittest

from location import locations_to_spans, spans_to_locations


class TestLocation(unittest.TestCase):

    def test_locations_to_spans(self):
        locations = ['461 467;483 489', '461 467;506 519']
        expected = [(461, 467), (483, 489), (461, 467), (506, 519)]
        self.assertEqual(locations_to_spans(locations), expected)

    def test_spans_to_locations(self):
        spans = [(461, 467), (483, 489), (461, 467), (506, 519)]
        expected = ['461 467; 483 489; 461 467; 506 519']
        self.assertEqual(spans_to_locations(spans), expected)

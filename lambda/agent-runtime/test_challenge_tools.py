import unittest
from challenge_tools import prime_count, count_tiles, validate_url, PageText


class ChallengeToolsTests(unittest.TestCase):
    def test_primes(self):
        self.assertEqual(prime_count(2, 1_000_000), 78498)
        self.assertEqual(prime_count(0, 10), 4)
        self.assertEqual(prime_count(11, 11), 1)
        self.assertEqual(prime_count(-5, 1), 0)
        self.assertEqual(prime_count(1_000_000, 1_010_000), sum(
            all(n % p for p in range(2, __import__('math').isqrt(n) + 1))
            for n in range(1_000_000, 1_010_001)))

    def test_limits(self):
        for interval in [(10, 2), (2, 100_000_001), (2, 2_000_003)]:
            with self.assertRaises(ValueError):
                prime_count(*interval)

    def test_map(self):
        self.assertEqual(count_tiles([['c5', 'c1'], ['C5', 'w']], 'c5'), 2)
        with self.assertRaises(ValueError):
            count_tiles(None, 'c5')

    def test_url(self):
        self.assertEqual(validate_url('https://aws.amazon.com/nova/forge/'), 'https://aws.amazon.com/nova/forge/')
        for url in ['http://aws.amazon.com/', 'https://localhost/', 'https://aws.amazon.com.evil.test/', 'https://user@aws.amazon.com/', 'https://aws.amazon.com:8080/']:
            with self.assertRaises(ValueError):
                validate_url(url)

    def test_page(self):
        parser = PageText()
        parser.feed('<p>Evidence</p><script>ignore()</script><style>ignore</style>Useful')
        self.assertEqual(parser.parts, ['Evidence', 'Useful'])


if __name__ == '__main__':
    unittest.main()

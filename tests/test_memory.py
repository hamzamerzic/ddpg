import unittest
import env
from ddpg import memory


class TestReplayBuffer(unittest.TestCase):

    def test_size(self):
        rb = memory.ReplayBuffer(2)
        self.assertEqual(len(rb), 0)
        rb.add(1)
        self.assertEqual(len(rb), 1)
        rb.add(2)
        rb.add(3)
        self.assertEqual(len(rb), 2)


if __name__ == '__main__':
    unittest.main()

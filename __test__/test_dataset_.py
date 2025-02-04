import unittest
import torch
from dataset import InpaintingDataset, InpaintingDatasetV2

class TestInpaintingDataset(unittest.TestCase):
    
    def setUp(self):
        self.data_root_dir = 'data'
        self.mask_root_dir = 'mask'
        self.num_frames = 10
        self.resize_shape = (256, 256)
        self.dataset = InpaintingDataset(self.data_root_dir, self.mask_root_dir, self.num_frames, self.resize_shape)
    
    def test_ds_length(self):
        self.assertEqual(len(self.dataset), 10)
    
    def test_getitem_valid_idx(self):
        idx = 0
        _, data, target = self.dataset[idx]
        expected_shape = (self.num_frames, 1, self.resize_shape[0], self.resize_shape[1])
        self.assertEqual(data.shape, expected_shape)
        self.assertEqual(target.shape, expected_shape)
        self.assertIsInstance(data, torch.Tensor)
        self.assertIsInstance(target, torch.Tensor)
    
    def test_getitem_invalid_idx(self):
        with self.assertRaises(IndexError):
            self.dataset[len(self.dataset)]
    
    def test_data_values(self):
        idx = 0
        _, data, target = self.dataset[idx]
        self.assertTrue(data.min() >= 0 and data.max() <= 1)
        self.assertTrue(target.min() >= 0 and target.max() <= 1)

class TestInpaintingDatasetV2(TestInpaintingDataset):
    
    def setUp(self):
        self.data_root_dir = 'data'
        self.mask_root_dir = 'mask'
        self.num_frames = 10
        self.resize_shape = (256, 256)
        self.dataset = InpaintingDatasetV2(self.data_root_dir, self.mask_root_dir, self.num_frames, self.resize_shape)
    
if __name__ == '__main__':
    unittest.main()
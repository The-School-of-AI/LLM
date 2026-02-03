"""
Tests for S3 Checkpoint Manager.

Run with:
    pytest test/test_checkpoint.py -v
    
Or test specific functions:
    pytest test/test_checkpoint.py::test_s3_config -v
"""

import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from aws.config import S3Config, get_default_config


class TestS3Config:
    """Test S3Config class."""
    
    def test_s3_config_creation(self):
        """Test basic S3Config creation."""
        config = S3Config(
            bucket_name='test-bucket',
            s3_prefix='test/prefix',
            region='us-west-2'
        )
        
        assert config.bucket_name == 'test-bucket'
        assert config.s3_prefix == 'test/prefix'
        assert config.region == 'us-west-2'
        assert config.keep_last_n_checkpoints == 3  # Default
    
    def test_s3_config_validation(self):
        """Test S3Config validation."""
        # Valid config
        config = S3Config(
            bucket_name='test-bucket',
            s3_prefix='test/prefix'
        )
        config.validate()  # Should not raise
        
        # Invalid: missing bucket name
        with pytest.raises(ValueError):
            config = S3Config(bucket_name='', s3_prefix='test')
            config.validate()
        
        # Invalid: missing prefix
        with pytest.raises(ValueError):
            config = S3Config(bucket_name='test', s3_prefix='')
            config.validate()
        
        # Invalid: keep_last_n < 1
        with pytest.raises(ValueError):
            config = S3Config(
                bucket_name='test',
                s3_prefix='test',
                keep_last_n_checkpoints=0
            )
            config.validate()
    
    def test_s3_config_from_env(self):
        """Test S3Config creation from environment variables."""
        with patch.dict(os.environ, {
            'S3_BUCKET_NAME': 'env-bucket',
            'S3_PREFIX': 'env/prefix',
            'S3_REGION': 'eu-west-1',
            'KEEP_LAST_N_CHECKPOINTS': '5'
        }):
            config = S3Config.from_env()
            
            assert config.bucket_name == 'env-bucket'
            assert config.s3_prefix == 'env/prefix'
            assert config.region == 'eu-west-1'
            assert config.keep_last_n_checkpoints == 5
    
    def test_s3_config_from_env_with_override(self):
        """Test S3Config from env with overrides."""
        with patch.dict(os.environ, {
            'S3_BUCKET_NAME': 'env-bucket',
            'S3_PREFIX': 'env/prefix',
        }):
            config = S3Config.from_env(
                bucket_name='override-bucket',
                verbose=False
            )
            
            assert config.bucket_name == 'override-bucket'  # Overridden
            assert config.s3_prefix == 'env/prefix'  # From env
            assert config.verbose == False  # Overridden
    
    def test_get_default_config(self):
        """Test preset configurations."""
        dev_config = get_default_config('development')
        assert dev_config.bucket_name == 'dev-training-checkpoints'
        assert dev_config.keep_last_n_checkpoints == 2
        
        prod_config = get_default_config('production')
        assert prod_config.bucket_name == 'prod-training-checkpoints'
        assert prod_config.keep_last_n_checkpoints == 5
        
        # Invalid preset
        with pytest.raises(ValueError):
            get_default_config('invalid-preset')
    
    def test_boto3_config_generation(self):
        """Test boto3 configuration generation."""
        config = S3Config(
            bucket_name='test-bucket',
            s3_prefix='test',
            region='us-east-1',
            aws_access_key_id='test-key',
            aws_secret_access_key='test-secret'
        )
        
        boto3_config = config.get_boto3_config()
        
        assert boto3_config['region_name'] == 'us-east-1'
        assert boto3_config['aws_access_key_id'] == 'test-key'
        assert boto3_config['aws_secret_access_key'] == 'test-secret'


class TestS3CheckpointManager:
    """Test S3CheckpointManager class."""
    
    @pytest.fixture
    def mock_s3_config(self):
        """Create a test S3 config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = S3Config(
                bucket_name='test-bucket',
                s3_prefix='test/checkpoints',
                local_checkpoint_dir=tmpdir,
                region='us-east-1',
                verbose=False
            )
            yield config
    
    @pytest.fixture
    def mock_distributed_env(self):
        """Mock distributed training environment."""
        with patch.dict(os.environ, {
            'LOCAL_RANK': '0',
            'RANK': '0',
            'WORLD_SIZE': '1'
        }):
            yield
    
    @patch('boto3.client')
    @patch('torch.distributed.is_initialized')
    def test_checkpoint_manager_initialization(
        self, 
        mock_dist_init,
        mock_boto_client,
        mock_s3_config,
        mock_distributed_env
    ):
        """Test checkpoint manager initialization."""
        from src.checkpoint import S3CheckpointManager
        
        mock_dist_init.return_value = False
        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client
        
        manager = S3CheckpointManager(mock_s3_config)
        
        assert manager.config == mock_s3_config
        assert manager.global_rank == 0
        assert manager.local_rank == 0
        assert manager.world_size == 1
        assert manager.is_global_main == True
        assert manager.is_local_main == True
    
    @patch('boto3.client')
    @patch('torch.distributed.is_initialized')
    def test_should_upload_file_single_node(
        self,
        mock_dist_init,
        mock_boto_client,
        mock_s3_config,
        mock_distributed_env
    ):
        """Test file upload decision in single-node setup."""
        from src.checkpoint import S3CheckpointManager
        
        mock_dist_init.return_value = False
        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client
        
        manager = S3CheckpointManager(mock_s3_config)
        
        # Single node should upload all files
        assert manager._should_upload_file('mp_rank_00_model_states.pt') == True
        assert manager._should_upload_file('latest') == True
        assert manager._should_upload_file('global_step.txt') == True
    
    @patch('boto3.client')
    @patch('torch.distributed.is_initialized')
    @patch('torch.cuda.device_count')
    def test_should_upload_file_multi_node(
        self,
        mock_device_count,
        mock_dist_init,
        mock_boto_client,
        mock_s3_config
    ):
        """Test file upload decision in multi-node setup."""
        from src.checkpoint import S3CheckpointManager
        
        # Simulate 2 nodes with 4 GPUs each
        with patch.dict(os.environ, {
            'LOCAL_RANK': '0',
            'RANK': '4',  # Second node, first GPU
            'WORLD_SIZE': '8'
        }):
            mock_device_count.return_value = 4
            mock_dist_init.return_value = False
            mock_s3_client = MagicMock()
            mock_boto_client.return_value = mock_s3_client
            
            manager = S3CheckpointManager(mock_s3_config)
            
            assert manager.num_nodes == 2
            assert manager.node_rank == 1  # Second node
            
            # Node 1 should upload its rank files (4-7)
            assert manager._should_upload_file('mp_rank_04_model_states.pt') == True
            assert manager._should_upload_file('mp_rank_05_model_states.pt') == True
            
            # Node 1 should NOT upload node 0's files
            assert manager._should_upload_file('mp_rank_00_model_states.pt') == False
            
            # Metadata files: only node 0 uploads
            assert manager._should_upload_file('latest') == False
            assert manager._should_upload_file('global_step.txt') == False


class TestCheckpointIntegration:
    """Integration tests (require mocking DeepSpeed)."""
    
    @pytest.fixture
    def mock_model_engine(self):
        """Create a mock DeepSpeed model engine."""
        engine = Mock()
        engine.save_checkpoint = Mock()
        engine.load_checkpoint = Mock(return_value=(None, {'step': 100}))
        return engine
    
    @patch('boto3.client')
    @patch('torch.distributed.is_initialized')
    @patch('torch.distributed.barrier')
    def test_save_checkpoint(
        self,
        mock_barrier,
        mock_dist_init,
        mock_boto_client,
        mock_model_engine
    ):
        """Test checkpoint saving."""
        from src.checkpoint import S3CheckpointManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {
                'LOCAL_RANK': '0',
                'RANK': '0',
                'WORLD_SIZE': '1'
            }):
                config = S3Config(
                    bucket_name='test-bucket',
                    s3_prefix='test',
                    local_checkpoint_dir=tmpdir,
                    verbose=False
                )
                
                mock_dist_init.return_value = False
                mock_s3_client = MagicMock()
                mock_boto_client.return_value = mock_s3_client
                
                manager = S3CheckpointManager(config)
                
                # Save checkpoint
                manager.save_checkpoint(
                    mock_model_engine,
                    step=100,
                    client_state={'epoch': 1}
                )
                
                # Verify DeepSpeed save was called
                mock_model_engine.save_checkpoint.assert_called_once()
                
                # Verify checkpoint was queued for upload
                assert 100 in manager.active_uploads


def test_checkpoint_config_repr():
    """Test config string representation."""
    config = S3Config(
        bucket_name='test-bucket',
        s3_prefix='test/prefix',
        region='us-west-2'
    )
    
    repr_str = repr(config)
    assert 'test-bucket' in repr_str
    assert 'test/prefix' in repr_str
    assert 'us-west-2' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import perform_experiments_bfts

class TestPerformExperiments(unittest.TestCase):
    @patch('ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.load_cfg')
    @patch('ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.load_task_desc')
    @patch('ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.prep_agent_workspace')
    @patch('ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.AgentManager')
    @patch('ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.Interpreter')
    @patch('ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.backend.compile_prompt_to_md')
    @patch('ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.shutil.rmtree')
    @patch('ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.save_run')
    @patch('ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.overall_summarize')
    @patch('ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.Live') # Mock Rich Live
    def test_perform_experiments_flow(self, mock_live, mock_summarize, mock_save_run, 
                                      mock_rmtree, mock_compile_prompt, mock_interpreter_cls, 
                                      mock_agent_manager_cls, mock_prep_workspace, 
                                      mock_load_task_desc, mock_load_cfg):
        
        # Setup Mocks
        mock_cfg = MagicMock()
        mock_cfg.exp_name = "test_experiment"
        mock_cfg.workspace_dir = "/tmp/test_workspace"
        mock_cfg.log_dir = Path("/tmp/test_log")
        mock_cfg.agent.steps = 10
        mock_cfg.generate_report = True
        
        # Ensure directories exist
        import os
        os.makedirs(mock_cfg.workspace_dir, exist_ok=True)
        os.makedirs(mock_cfg.log_dir, exist_ok=True)

        mock_load_cfg.return_value = mock_cfg
        mock_load_task_desc.return_value = {"Title": "Test Task"}
        mock_compile_prompt.return_value = "# Test Task"
        
        # Mock Interpreter
        mock_interpreter_instance = mock_interpreter_cls.return_value
        mock_interpreter_instance.run.return_value = ("output", "error")
        
        # Mock Agent Manager
        mock_manager_instance = mock_agent_manager_cls.return_value
        mock_manager_instance.journals = {}
        mock_manager_instance.completed_stages = []
        mock_manager_instance.current_stage = MagicMock()
        mock_manager_instance.current_stage.name = "Initial"
        
        # Mock Summarize
        mock_summarize.return_value = ({}, {}, {}, {})
        
        # Run the function
        perform_experiments_bfts("dummy_config.yaml")
        
        # Verifications
        mock_load_cfg.assert_called_once()
        mock_prep_workspace.assert_called_once_with(mock_cfg)
        mock_interpreter_cls.assert_called_once() # Ensure interpreter is initialized
        mock_agent_manager_cls.assert_called_once()
        
        # Verify manager.run was called
        mock_manager_instance.run.assert_called_once()
        
        # Verify summarize was called (since generate_report is True)
        mock_summarize.assert_called_once()

    @patch('ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.load_cfg')
    @patch('ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.load_task_desc')
    @patch('ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.prep_agent_workspace')
    @patch('ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.AgentManager')
    @patch('ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.Interpreter')
    @patch('ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.backend.compile_prompt_to_md')
    @patch('ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.shutil.rmtree')
    @patch('ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.Live')
    def test_cleanup_on_failure(self, mock_live, mock_rmtree, mock_compile_prompt, 
                                mock_interpreter_cls, mock_agent_manager_cls, 
                                mock_prep_workspace, mock_load_task_desc, mock_load_cfg):
        
        # Setup Mocks to simulate early failure
        mock_cfg = MagicMock()
        mock_cfg.workspace_dir = "/tmp/test_workspace"
        mock_load_cfg.return_value = mock_cfg
        
        # Simulate failure in prep_agent_workspace
        mock_prep_workspace.side_effect = Exception("Setup failed")
        
        with self.assertRaises(Exception):
            perform_experiments_bfts("dummy_config.yaml")
            
        # Verify clean up logic might be triggered if we manually called cleanup, 
        # but atexit is hard to test in unittest without invoking it.
        # However, we can verify that flow stopped.

if __name__ == '__main__':
    unittest.main()

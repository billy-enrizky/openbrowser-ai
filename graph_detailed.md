graph TB
    subgraph "Browser-Use LangGraph Execution Flow"
        START([START]) --> prepare_context
        
        subgraph "Step Execution Loop"
            prepare_context["`**prepare_context_node**
            • Get browser state summary
            • Update action models
            • Create state messages
            • Check stop/pause conditions`"]
            
            get_next_action["`**get_next_action_node**
            • Get messages from message manager
            • Call LLM with retry logic
            • Handle post-LLM processing
            • Store model output`"]
            
            execute_actions["`**execute_actions_node**
            • Extract actions from model output
            • Execute each action via tools.act()
            • Collect ActionResult objects
            • Handle action errors`"]
            
            post_process["`**post_process_node**
            • Check for new downloads
            • Update consecutive failures counter
            • Log completion results
            • Reset failure counter on success`"]
            
            finalize["`**finalize_node**
            • Create history item
            • Store screenshot
            • Log step completion
            • Save file system state
            • Emit CreateAgentStepEvent
            • Increment step counter`"]
        end
        
        prepare_context --> get_next_action
        get_next_action --> execute_actions
        execute_actions --> post_process
        post_process --> finalize
        
        subgraph "Conditional Routing"
            should_continue{"`**should_continue()**
            Check:
            • stopped flag?
            • paused flag?
            • is_done?
            • max_steps reached?
            • max_failures reached?`"}
            
            should_continue -->|continue<br/>All checks pass| prepare_context
            should_continue -->|stop<br/>Any check fails| END([END])
        end
        
        finalize --> should_continue
        
        subgraph "Error Handling"
            handle_error["`**handle_error_node**
            • Format error message
            • Update consecutive failures
            • Create error ActionResult
            • Log error details`"]
            
            prepare_context -.->|exception| handle_error
            get_next_action -.->|exception| handle_error
            execute_actions -.->|exception| handle_error
            post_process -.->|exception| handle_error
            
            handle_error --> finalize
        end
    end
    
    %% Styling
    classDef startEnd fill:#4caf50,stroke:#2e7d32,stroke-width:3px,color:#fff
    classDef processNode fill:#2196f3,stroke:#1565c0,stroke-width:2px,color:#fff
    classDef decisionNode fill:#ff9800,stroke:#e65100,stroke-width:2px,color:#fff
    classDef errorNode fill:#f44336,stroke:#c62828,stroke-width:2px,color:#fff
    
    class START,END startEnd
    class prepare_context,get_next_action,execute_actions,post_process,finalize processNode
    class should_continue decisionNode
    class handle_error errorNode


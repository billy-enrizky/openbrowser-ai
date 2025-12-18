graph TD
    START([START]) --> prepare_context[prepare_context<br/>Prepare browser state and context]
    
    prepare_context --> get_next_action[get_next_action<br/>Get next action from LLM]
    
    get_next_action --> execute_actions[execute_actions<br/>Execute actions via tools]
    
    execute_actions --> post_process[post_process<br/>Handle post-action processing]
    
    post_process --> finalize[finalize<br/>Finalize step with history]
    
    finalize --> should_continue{should_continue?<br/>Check conditions}
    
    should_continue -->|continue| prepare_context
    should_continue -->|stop| END([END])
    
    %% Error handling path
    prepare_context -.->|error| handle_error[handle_error<br/>Handle errors]
    get_next_action -.->|error| handle_error
    execute_actions -.->|error| handle_error
    post_process -.->|error| handle_error
    
    handle_error --> finalize
    
    %% Styling
    classDef startEnd fill:#e1f5e1,stroke:#4caf50,stroke-width:3px
    classDef processNode fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    classDef decisionNode fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    classDef errorNode fill:#ffebee,stroke:#f44336,stroke-width:2px
    
    class START,END startEnd
    class prepare_context,get_next_action,execute_actions,post_process,finalize processNode
    class should_continue decisionNode
    class handle_error errorNode


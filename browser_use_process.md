# Browser-Use Agent Process Flow

This document explains the complete process flow of the browser-use agent, including CDP integration, page extraction, screenshot decisions, and memory management.

```mermaid
flowchart TB
    Start([Agent.run starts]) --> Init[Initialize Agent]
    Init --> BrowserInit[Initialize BrowserSession]
    BrowserInit --> CDPConnect{CDP Connection}
    
    CDPConnect -->|Local Browser| StartLocal[Start Chromium Process]
    CDPConnect -->|Remote CDP| ConnectRemote[Connect to CDP URL]
    CDPConnect -->|Cloud Browser| ProvisionCloud[Provision Cloud Browser]
    
    StartLocal --> CDPClient[Create CDP Client]
    ConnectRemote --> CDPClient
    ProvisionCloud --> CDPClient
    
    CDPClient --> SessionMgr[Initialize SessionManager]
    SessionMgr --> AutoAttach[Enable Auto-Attach for Targets]
    AutoAttach --> Watchdogs[Start Watchdog Services]
    
    Watchdogs --> DOMWatchdog[DOMWatchdog<br/>DOM snapshots & screenshots]
    Watchdogs --> PopupsWatchdog[PopupsWatchdog<br/>JavaScript dialogs]
    Watchdogs --> DownloadsWatchdog[DownloadsWatchdog<br/>File downloads]
    Watchdogs --> SecurityWatchdog[SecurityWatchdog<br/>Domain restrictions]
    Watchdogs --> AboutBlankWatchdog[AboutBlankWatchdog<br/>Empty page redirects]
    
    DOMWatchdog --> EventBus[Event Bus<br/>bubus]
    PopupsWatchdog --> EventBus
    DownloadsWatchdog --> EventBus
    SecurityWatchdog --> EventBus
    AboutBlankWatchdog --> EventBus
    
    EventBus --> StepLoop[Step Loop<br/>max_steps iterations]
    
    StepLoop --> PrepareContext[Prepare Context]
    
    PrepareContext --> GetBrowserState[Get Browser State Summary]
    
    GetBrowserState --> CDPGetDOM[CDP: DOM.getDocument]
    CDPGetDOM --> CDPGetNodes[CDP: DOM.getNodes]
    CDPGetNodes --> CDPGetAXTree[CDP: Accessibility.getFullAXTree]
    CDPGetAXTree --> CDPGetSnapshot[CDP: DOMSnapshot.captureSnapshot]
    
    CDPGetSnapshot --> BuildDOMTree[Build Enhanced DOM Tree]
    BuildDOMTree --> SerializeDOM[Serialize DOM for LLM]
    SerializeDOM --> DOMState[SerializedDOMState<br/>with selector_map]
    
    DOMState --> CheckScreenshot{use_vision?}
    
    CheckScreenshot -->|True| AlwaysScreenshot[Always Include Screenshot]
    CheckScreenshot -->|'auto'| CheckActionRequest{Action Requested<br/>Screenshot?}
    CheckScreenshot -->|False| NoScreenshot[No Screenshot]
    
    CheckActionRequest -->|Yes| IncludeScreenshot[Include Screenshot]
    CheckActionRequest -->|No| NoScreenshot
    
    AlwaysScreenshot --> CDPScreenshot[CDP: Page.captureScreenshot]
    IncludeScreenshot --> CDPScreenshot
    CDPScreenshot --> ScreenshotBase64[Base64 Screenshot]
    
    ScreenshotBase64 --> CreateStateMessage[Create State Message]
    NoScreenshot --> CreateStateMessage
    
    CreateStateMessage --> MessageManager[MessageManager]
    
    MessageManager --> GetHistory[Get Message History]
    GetHistory --> SystemMsg[System Message<br/>Task & Instructions]
    SystemMsg --> StateMsg[State Message<br/>DOM + Screenshot]
    StateMsg --> ContextMsg[Context Messages<br/>Previous Steps]
    
    ContextMsg --> BuildMessages[Build LLM Messages]
    BuildMessages --> LLMCall[Call LLM]
    
    LLMCall --> ParseOutput[Parse AgentOutput]
    ParseOutput --> ExtractActions[Extract Actions]
    
    ExtractActions --> ExecuteActions[Execute Actions]
    
    ExecuteActions --> ActionLoop{For each action}
    
    ActionLoop --> GetActionType{Action Type?}
    
    GetActionType -->|click_element| CDPClick[CDP: Input.dispatchMouseEvent]
    GetActionType -->|type| CDPType[CDP: Input.insertText]
    GetActionType -->|scroll| CDPScroll[CDP: Input.dispatchMouseEvent<br/>+ scroll]
    GetActionType -->|navigate| CDPNavigate[CDP: Page.navigate]
    GetActionType -->|screenshot| CDPScreenshotAction[CDP: Page.captureScreenshot]
    GetActionType -->|done| DoneAction[Mark as Done]
    
    CDPClick --> WaitForLoad[Wait for Page Load]
    CDPType --> WaitForLoad
    CDPScroll --> WaitForLoad
    CDPNavigate --> WaitForLoad
    CDPScreenshotAction --> WaitForLoad
    DoneAction --> WaitForLoad
    
    WaitForLoad --> CollectResult[Collect ActionResult]
    CollectResult --> ActionLoop
    
    ActionLoop -->|More actions| GetActionType
    ActionLoop -->|All done| PostProcess[Post Process]
    
    PostProcess --> CheckDownloads[Check for Downloads]
    CheckDownloads --> UpdateFailures[Update Consecutive Failures]
    UpdateFailures --> LogResults[Log Results]
    
    LogResults --> Finalize[Finalize Step]
    
    Finalize --> CreateHistory[Create AgentHistory]
    CreateHistory --> StoreScreenshot[Store Screenshot]
    StoreScreenshot --> EmitEvent[Emit CreateAgentStepEvent]
    EmitEvent --> SaveFileSystem[Save File System State]
    
    SaveFileSystem --> UpdateMemory[Update Memory]
    
    UpdateMemory --> MemoryManager[MessageManager Memory]
    
    MemoryManager --> LongTermMemory{Long Term<br/>Memory?}
    LongTermMemory -->|Yes| AddToHistory[Add to Agent History]
    LongTermMemory -->|No| CheckExtracted{Extracted<br/>Content?}
    
    CheckExtracted -->|include_extracted_content_only_once| ReadState[Add to read_state<br/>One-time use]
    CheckExtracted -->|Regular| AddToHistory
    
    AddToHistory --> UpdateContext[Update Context Messages]
    ReadState --> UpdateContext
    
    UpdateContext --> CheckContinue{Should Continue?}
    
    CheckContinue -->|Stopped| End([End])
    CheckContinue -->|Paused| End
    CheckContinue -->|Done| End
    CheckContinue -->|Max Steps| End
    CheckContinue -->|Max Failures| End
    CheckContinue -->|Continue| StepLoop
    
    style Start fill:#e1f5ff
    style End fill:#ffe1f5
    style CDPClient fill:#fff4e1
    style EventBus fill:#e1ffe1
    style LLMCall fill:#f4e1ff
    style ExecuteActions fill:#ffe1e1
    style MemoryManager fill:#e1e1ff
```

## Detailed Component Explanations

### CDP (Chrome DevTools Protocol) Integration

```mermaid
flowchart LR
    CDP[CDP Client<br/>cdp-use library] --> Connect[Connect to Browser]
    Connect --> Local[Local Browser<br/>--remote-debugging-port]
    Connect --> Remote[Remote CDP URL<br/>ws://host:port]
    Connect --> Cloud[Cloud Browser<br/>Browser-Use Cloud]
    
    CDP --> SessionMgr[SessionManager]
    SessionMgr --> AutoAttach[Auto-Attach to Targets]
    AutoAttach --> PageTargets[Page Targets]
    AutoAttach --> WorkerTargets[Worker Targets]
    
    PageTargets --> CDPCalls[CDP Method Calls]
    
    CDPCalls --> DOM[DOM Domain<br/>getDocument, getNodes]
    CDPCalls --> Page[Page Domain<br/>navigate, captureScreenshot]
    CDPCalls --> Input[Input Domain<br/>dispatchMouseEvent, insertText]
    CDPCalls --> Accessibility[Accessibility Domain<br/>getFullAXTree]
    CDPCalls --> DOMSnapshot[DOMSnapshot Domain<br/>captureSnapshot]
    
    DOM --> DOMTree[Enhanced DOM Tree]
    Page --> Screenshot[Screenshot]
    Input --> Interactions[User Interactions]
    Accessibility --> AXTree[Accessibility Tree]
    DOMSnapshot --> Snapshot[DOM Snapshot]
    
    DOMTree --> Serialize[Serialize for LLM]
    AXTree --> Serialize
    Snapshot --> Serialize
```

### Page Extraction Process

```mermaid
flowchart TD
    StartExtract[Start Page Extraction] --> GetCurrentPage[Get Current Page Target]
    GetCurrentPage --> CDPDOM[CDP: DOM.getDocument]
    
    CDPDOM --> CDPNodes[CDP: DOM.getNodes<br/>Get all node IDs]
    CDPNodes --> CDPAttributes[CDP: DOM.getAttributes<br/>For each node]
    CDPAttributes --> CDPBoxModel[CDP: DOM.getBoxModel<br/>Get layout info]
    
    CDPBoxModel --> CDPAXTree[CDP: Accessibility.getFullAXTree<br/>Get accessibility info]
    CDPAXTree --> CDPSnapshot[CDP: DOMSnapshot.captureSnapshot<br/>Get computed styles]
    
    CDPSnapshot --> BuildTree[Build Enhanced DOM Tree]
    
    BuildTree --> ProcessNodes[Process Each Node]
    ProcessNodes --> ExtractText[Extract Text Content]
    ProcessNodes --> ExtractAttributes[Extract Attributes]
    ProcessNodes --> ExtractStyles[Extract Computed Styles]
    ProcessNodes --> ExtractLayout[Extract Layout Info]
    ProcessNodes --> ExtractAX[Extract Accessibility Info]
    
    ExtractText --> CreateNode[Create EnhancedDOMTreeNode]
    ExtractAttributes --> CreateNode
    ExtractStyles --> CreateNode
    ExtractLayout --> CreateNode
    ExtractAX --> CreateNode
    
    CreateNode --> HandleShadowDOM{Has Shadow DOM?}
    HandleShadowDOM -->|Yes| CDPShadowRoot[CDP: DOM.describeNode<br/>Get shadow root]
    CDPShadowRoot --> ProcessShadowNodes[Process Shadow Nodes]
    ProcessShadowNodes --> CreateNode
    HandleShadowDOM -->|No| HandleIframes
    
    HandleShadowDOM --> HandleIframes{Has Iframes?}
    HandleIframes -->|Yes| CDPIframes[CDP: Target.getTargets<br/>Get iframe targets]
    CDPIframes --> ProcessIframes[Process Iframe Documents]
    ProcessIframes --> BuildTree
    HandleIframes -->|No| SerializeDOM
    
    HandleIframes --> SerializeDOM[Serialize DOM Tree]
    
    SerializeDOM --> FilterInteractive[Filter Interactive Elements]
    FilterInteractive --> AssignIndexes[Assign Indexes<br/>for interaction]
    AssignIndexes --> CreateSelectorMap[Create Selector Map<br/>index → node]
    
    CreateSelectorMap --> SerializeForLLM[Serialize for LLM]
    SerializeForLLM --> HTMLSerializer[HTML Serializer]
    HTMLSerializer --> MarkdownConverter[Markdown Converter]
    MarkdownConverter --> LLMRepresentation[LLM Representation<br/>with indexes]
    
    LLMRepresentation --> BrowserStateSummary[BrowserStateSummary]
    BrowserStateSummary --> DoneExtract[Extraction Complete]
```

### Screenshot Decision Logic

```mermaid
flowchart TD
    StartScreenshot[Start Screenshot Decision] --> CheckUseVision{use_vision<br/>Setting?}
    
    CheckUseVision -->|True| AlwaysInclude[Always Include Screenshot]
    CheckUseVision -->|'auto'| CheckActionRequest{Action Result<br/>Requests Screenshot?}
    CheckUseVision -->|False| NeverInclude[Never Include Screenshot]
    
    CheckActionRequest -->|Yes| IncludeScreenshot[Include Screenshot]
    CheckActionRequest -->|No| NeverInclude
    
    AlwaysInclude --> CheckScreenshotExists{Screenshot<br/>Available?}
    IncludeScreenshot --> CheckScreenshotExists
    
    CheckScreenshotExists -->|Yes| ProcessScreenshot[Process Screenshot]
    CheckScreenshotExists -->|No| NoScreenshot[No Screenshot]
    
    ProcessScreenshot --> ResizeCheck{llm_screenshot_size<br/>Configured?}
    ResizeCheck -->|Yes| ResizeScreenshot[Resize Screenshot]
    ResizeCheck -->|No| Base64Encode[Base64 Encode]
    
    ResizeScreenshot --> Base64Encode
    Base64Encode --> VisionDetail{vision_detail_level?}
    
    VisionDetail -->|'auto'| AutoDetail[Auto Detail Level]
    VisionDetail -->|'low'| LowDetail[Low Detail Level]
    VisionDetail -->|'high'| HighDetail[High Detail Level]
    
    AutoDetail --> CreateImageParam[Create ContentPartImageParam]
    LowDetail --> CreateImageParam
    HighDetail --> CreateImageParam
    
    CreateImageParam --> AddToMessage[Add to User Message]
    AddToMessage --> LLMWithVision[Send to LLM with Vision]
    
    NeverInclude --> TextOnly[Text Only Message]
    NoScreenshot --> TextOnly
    TextOnly --> LLMTextOnly[Send to LLM Text Only]
```

### Memory Management

```mermaid
flowchart TD
    StartMemory[Action Result Received] --> CheckMemory{Has<br/>long_term_memory?}
    
    CheckMemory -->|Yes| AddToLongTerm[Add to Long Term Memory]
    CheckMemory -->|No| CheckExtracted{Has<br/>extracted_content?}
    
    AddToLongTerm --> UpdateHistory[Update Agent History]
    
    CheckExtracted -->|Yes| CheckIncludeOnce{include_extracted_content<br/>_only_once?}
    CheckExtracted -->|No| CheckError{Has Error?}
    
    CheckIncludeOnce -->|Yes| AddToReadState[Add to read_state<br/>One-time inclusion]
    CheckIncludeOnce -->|No| AddToLongTerm
    
    CheckError -->|Yes| AddErrorToHistory[Add Error to History]
    CheckError -->|No| CheckImages{Has Images?}
    
    AddErrorToHistory --> UpdateHistory
    CheckImages -->|Yes| AddToReadStateImages[Add to read_state_images<br/>One-time inclusion]
    CheckImages -->|No| UpdateHistory
    
    AddToReadState --> CheckReadStateSize{read_state<br/>> 60KB?}
    CheckReadStateSize -->|Yes| TruncateReadState[Truncate to 60KB]
    CheckReadStateSize -->|No| UpdateHistory
    
    AddToReadStateImages --> UpdateHistory
    
    UpdateHistory --> BuildContextMessages[Build Context Messages]
    
    BuildContextMessages --> CheckMaxHistory{max_history_items<br/>Configured?}
    CheckMaxHistory -->|Yes| LimitHistory[Limit to max_history_items<br/>Keep first + last N]
    CheckMaxHistory -->|No| FullHistory[Include Full History]
    
    LimitHistory --> FormatHistory[Format History Items]
    FullHistory --> FormatHistory
    
    FormatHistory --> AddToContext[Add to Context Messages]
    AddToContext --> NextStep[Ready for Next Step]
    
    NextStep --> ClearReadState[Clear read_state after use]
    ClearReadState --> ClearReadStateImages[Clear read_state_images after use]
    
    style AddToLongTerm fill:#e1ffe1
    style AddToReadState fill:#fff4e1
    style UpdateHistory fill:#e1e1ff
    style LimitHistory fill:#ffe1e1
```

### Complete Agent Step Flow

```mermaid
sequenceDiagram
    participant Agent
    participant BrowserSession
    participant CDP
    participant DomService
    participant MessageManager
    participant LLM
    participant Tools
    
    Agent->>BrowserSession: Get Browser State Summary
    BrowserSession->>CDP: DOM.getDocument()
    CDP-->>BrowserSession: Document Node
    BrowserSession->>CDP: DOM.getNodes()
    CDP-->>BrowserSession: All Node IDs
    BrowserSession->>CDP: Accessibility.getFullAXTree()
    CDP-->>BrowserSession: Accessibility Tree
    BrowserSession->>CDP: DOMSnapshot.captureSnapshot()
    CDP-->>BrowserSession: DOM Snapshot
    
    BrowserSession->>DomService: Build Enhanced DOM Tree
    DomService->>DomService: Process Nodes
    DomService->>DomService: Handle Shadow DOM
    DomService->>DomService: Handle Iframes
    DomService-->>BrowserSession: Enhanced DOM Tree
    
    BrowserSession->>DomService: Serialize DOM
    DomService->>DomService: Filter Interactive Elements
    DomService->>DomService: Assign Indexes
    DomService-->>BrowserSession: SerializedDOMState
    
    BrowserSession->>BrowserSession: Check use_vision
    alt use_vision == True or action requested
        BrowserSession->>CDP: Page.captureScreenshot()
        CDP-->>BrowserSession: Screenshot (base64)
    end
    
    BrowserSession-->>Agent: BrowserStateSummary
    
    Agent->>MessageManager: Create State Messages
    MessageManager->>MessageManager: Get Message History
    MessageManager->>MessageManager: Build State Message
    MessageManager-->>Agent: Messages Ready
    
    Agent->>LLM: Send Messages
    LLM-->>Agent: AgentOutput
    
    Agent->>Agent: Parse Actions
    Agent->>Tools: Execute Actions
    
    loop For each action
        Tools->>CDP: Execute Action (click, type, etc.)
        CDP-->>Tools: ActionResult
    end
    
    Tools-->>Agent: All Action Results
    
    Agent->>MessageManager: Update Memory
    MessageManager->>MessageManager: Process long_term_memory
    MessageManager->>MessageManager: Process extracted_content
    MessageManager->>MessageManager: Update History
    
    Agent->>Agent: Check Continue
    alt Should Continue
        Agent->>Agent: Next Step
    else Should Stop
        Agent->>Agent: End
    end
```

## Key Concepts

### 1. CDP Integration
- **Connection Types**: Local browser, Remote CDP URL, or Cloud Browser
- **Session Management**: Auto-attach to all targets (pages, workers)
- **Event-Driven**: Watchdog services monitor browser events via event bus
- **Direct CDP Calls**: DOM, Page, Input, Accessibility, DOMSnapshot domains

### 2. Page Extraction
- **Multi-Step Process**: Get document → Get nodes → Get attributes → Get layout → Get accessibility
- **Enhanced DOM Tree**: Combines DOM, accessibility, and layout information
- **Shadow DOM Support**: Recursively processes shadow roots
- **Iframe Support**: Handles cross-origin iframes
- **Serialization**: Converts to LLM-friendly format with interactive element indexes

### 3. Screenshot Decision
- **use_vision=True**: Always include screenshot
- **use_vision='auto'**: Include only if action requests it
- **use_vision=False**: Never include screenshot
- **Vision Detail Levels**: 'auto', 'low', or 'high' for image processing
- **Screenshot Resizing**: Optional resizing for LLM efficiency

### 4. Memory Management
- **Long Term Memory**: Stored in agent history, persists across steps
- **Read State**: One-time inclusion in next message (for large content)
- **Read State Images**: One-time image inclusion
- **History Limiting**: Optional max_history_items to control context size
- **Error Handling**: Errors added to history for learning

### 5. Agent Execution Flow
- **Step Loop**: Iterates up to max_steps
- **Context Preparation**: Gets browser state, builds messages
- **LLM Interaction**: Sends messages, receives actions
- **Action Execution**: Executes actions via CDP
- **Post Processing**: Updates memory, checks for completion
- **Continuation Logic**: Checks stop conditions (done, max steps, max failures)


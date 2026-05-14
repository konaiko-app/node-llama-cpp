import {ChatHistoryItem, ChatUserContentPart, ChatUserMessage} from "../types.js";

/**
 * Appends a user message to the chat history.
 * If the last message in the chat history is also a user message, the new message will be appended to it.
 */
export function appendUserMessageToChatHistory(
    chatHistory: readonly ChatHistoryItem[],
    message: string | ChatUserContentPart[]
) {
    const newChatHistory = chatHistory.slice();

    if (newChatHistory.length > 0 && newChatHistory[newChatHistory.length - 1]!.type === "user") {
        const lastUserMessage = newChatHistory[newChatHistory.length - 1]! as ChatUserMessage;

        newChatHistory[newChatHistory.length - 1] = {
            ...lastUserMessage,
            text: mergeUserTexts(lastUserMessage.text, message)
        };
    } else {
        newChatHistory.push({
            type: "user",
            text: message
        });
    }

    return newChatHistory;
}

function mergeUserTexts(
    existing: string | ChatUserContentPart[],
    incoming: string | ChatUserContentPart[]
): string | ChatUserContentPart[] {
    // Both strings: join with newline (original behavior)
    if (typeof existing === "string" && typeof incoming === "string")
        return [existing, incoming].join("\n\n");

    // Convert to content parts and merge
    const existingParts = typeof existing === "string"
        ? [{type: "text" as const, text: existing}]
        : existing;
    const incomingParts = typeof incoming === "string"
        ? [{type: "text" as const, text: incoming}]
        : incoming;

    return [...existingParts, {type: "text" as const, text: "\n\n"}, ...incomingParts];
}

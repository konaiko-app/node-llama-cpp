import {MtmdBitmapInput} from "../bindings/AddonTypes.js";
import {
    ChatHistoryItem, ChatImageInput, ChatModelResponse, ChatSystemMessage,
    ChatUserContentPart
} from "../types.js";

export const MEDIA_MARKER = "<__media__>";

export type MmprojChatHistoryItem =
    | ChatSystemMessage
    | {type: "user", text: string | ChatUserContentPart[]}
    | ChatModelResponse;

export function imageToBitmapInput(image: ChatImageInput): MtmdBitmapInput {
    if ("fileData" in image)
        return {fileData: image.fileData};

    return {data: image.data, width: image.width, height: image.height};
}

export function normalizeHistory(history: MmprojChatHistoryItem[]): {
    chatHistory: ChatHistoryItem[],
    images: MtmdBitmapInput[]
} {
    const images: MtmdBitmapInput[] = [];
    const chatHistory: ChatHistoryItem[] = history.map((item) => {
        if (item.type !== "user")
            return item as ChatHistoryItem;

        // Already-normalized string messages may contain leftover <__media__> markers
        // from previous turns whose image data has already been consumed.
        // Replace them with a neutral placeholder so they don't cause a
        // bitmap-count mismatch in mtmd_tokenize.
        if (typeof item.text === "string") {
            if (item.text.includes(MEDIA_MARKER))
                return {type: "user" as const, text: item.text.replaceAll(MEDIA_MARKER, "[image]")};

            return item as ChatHistoryItem;
        }

        let text = "";
        for (const part of item.text) {
            if (part.type === "text")
                text += part.text;
            else if (part.type === "image") {
                text += MEDIA_MARKER;
                images.push(imageToBitmapInput(part.image));
            }
        }

        return {type: "user" as const, text};
    });

    return {chatHistory, images};
}

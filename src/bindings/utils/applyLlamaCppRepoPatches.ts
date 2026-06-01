import path from "path";
import fs from "fs-extra";
import {simpleGit} from "simple-git";
import {llamaCppDirectory, llamaCppPatchesDirectory} from "../../config.js";
import {getConsoleLogPrefix} from "../../utils/getConsoleLogPrefix.js";

type RepoPatch = {
    filename: string,
    title: string,
    canSkip(repoPath: string, lastCommitDate?: Date): Promise<boolean>
};

// No upstream patches: konaiko-app/llama.cpp already carries everything the addon
// needs (incl. the MTP engine). PR-22566 was dropped — it doesn't apply on top of the
// MTP changes to llama-model-loader.cpp, and the no_alloc memory-breakdown fix it
// added is non-critical.
const patches: RepoPatch[] = [];

export function hasLlamaCppRepoPatchesToApply() {
    return patches.length > 0;
}

export async function applyLlamaCppRepoPatches(lastCommitDate?: Date, throwOnError: boolean = false) {
    if (!hasLlamaCppRepoPatchesToApply())
        return;

    if (!(await fs.pathExists(llamaCppPatchesDirectory)) || !(await fs.pathExists(llamaCppDirectory)))
        return;

    const git = simpleGit({baseDir: llamaCppDirectory});
    for (const patch of patches) {
        const patchPath = path.join(path.resolve(llamaCppPatchesDirectory), patch.filename);

        try {
            if (!(await fs.pathExists(patchPath))) {
                console.warn(`Patch file "${patch.filename}" not found, skipping patch "${patch.title}"`);
                continue;
            }

            if (await patch.canSkip(llamaCppDirectory, lastCommitDate))
                continue;
        } catch (err) {
            console.warn(
                getConsoleLogPrefix(),
                `Failed testing whether patch "${patch.filename}": "${patch.title}" can be skipped:`,
                String(err)
            );
        }

        try {
            await git.applyPatch(patchPath, {"--ignore-whitespace": null});
        } catch (err) {
            console.error(
                getConsoleLogPrefix(),
                `Failed to apply patch "${patch.filename}": "${patch.title}", building llama.cpp may fail.`,
                String(err)
            );

            if (throwOnError)
                throw err;
        }
    }
}

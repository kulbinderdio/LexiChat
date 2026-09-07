import { useState } from "react";
import { Plus, Trash2, PanelLeftClose, Pin } from "lucide-react";

// Mirrors the Rust `history::ConversationMeta` (snake_case over the wire).
export interface ConversationMeta {
  id: string;
  title: string;
  profile_id: string | null;
  model: string;
  created_at: number; // unix seconds
  updated_at: number;
  message_count: number;
  /// Pinned chats are returned first by the backend and shown above a divider in the list.
  pinned?: boolean;
  /// On-disk footprint: the saved JSON plus any working files kept with the chat. Computed by the
  /// backend on each list, so it reflects reality rather than a stored guess.
  size_bytes?: number;
}

interface Props {
  visible: boolean;
  conversations: ConversationMeta[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onPin: (id: string, pinned: boolean) => void;
  onHide: () => void;
}

// Chats now carry their working files (offloaded tool results, /work/artifacts data), so a
// research-heavy one can be materially larger than a chatty one. Surfacing the size is what makes
// that manageable rather than mysterious.
function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / 1048576).toFixed(bytes >= 10 * 1048576 ? 0 : 1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}
const LARGE_BYTES = 5 * 1024 * 1024;   // worth a second look when tidying up

function relativeTime(unixSecs: number): string {
  const diff = Date.now() / 1000 - unixSecs;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
  return new Date(unixSecs * 1000).toLocaleDateString();
}

export function HistoryPanel({ visible, conversations, activeId, onSelect, onNew, onDelete, onRename, onPin, onHide }: Props) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  if (!visible) return null;

  const startRename = (c: ConversationMeta) => {
    setEditingId(c.id);
    setDraft(c.title);
  };

  const commitRename = () => {
    if (editingId) {
      const title = draft.trim();
      if (title) onRename(editingId, title);
    }
    setEditingId(null);
  };

  return (
    <div className="history-panel">
      <div className="history-header">
        <button className="history-collapse" onClick={onHide} title="Hide history">
          <PanelLeftClose size={15} />
        </button>
        <span>Chat history</span>
        <button className="history-new" onClick={onNew} title="New chat">
          <Plus size={13} /> New
        </button>
      </div>

      <div className="history-list">
        {conversations.length === 0 && (
          <div className="history-empty">No saved conversations yet.</div>
        )}
        {conversations.map((c, i) => (
          <div key={c.id} className="history-row">
            {/* Divider between the pinned group and the rest — only when both exist. */}
            {!c.pinned && i > 0 && conversations[i - 1].pinned && (
              <div className="history-divider" />
            )}
          <div
            className={`history-item ${c.id === activeId ? "active" : ""} ${c.pinned ? "pinned" : ""}`}
            onClick={() => editingId !== c.id && onSelect(c.id)}
          >
            <div className="history-item-main">
              {editingId === c.id ? (
                <input
                  className="history-rename-input"
                  autoFocus
                  value={draft}
                  onChange={e => setDraft(e.target.value)}
                  onClick={e => e.stopPropagation()}
                  onBlur={commitRename}
                  onKeyDown={e => {
                    if (e.key === "Enter") commitRename();
                    if (e.key === "Escape") setEditingId(null);
                  }}
                />
              ) : (
                <div
                  className="history-title"
                  title={c.title}
                  onDoubleClick={e => { e.stopPropagation(); startRename(c); }}
                >
                  {c.title}
                </div>
              )}
              <div className="history-meta">
                {relativeTime(c.updated_at)}
                {c.size_bytes != null && c.size_bytes > 0 && (
                  <>
                    {" · "}
                    <span
                      className={c.size_bytes >= LARGE_BYTES ? "history-size large" : "history-size"}
                      title={`${c.size_bytes.toLocaleString()} bytes on disk, including this chat's working files`}
                    >
                      {formatSize(c.size_bytes)}
                    </span>
                  </>
                )}
              </div>
            </div>
            <button
              className={`history-pin ${c.pinned ? "on" : ""}`}
              title={c.pinned ? "Unpin conversation" : "Pin conversation"}
              onClick={e => { e.stopPropagation(); onPin(c.id, !c.pinned); }}
            >
              <Pin size={13} />
            </button>
            <button
              className="history-del"
              title="Delete conversation"
              onClick={e => { e.stopPropagation(); onDelete(c.id); }}
            >
              <Trash2 size={13} />
            </button>
          </div>
          </div>
        ))}
      </div>
    </div>
  );
}

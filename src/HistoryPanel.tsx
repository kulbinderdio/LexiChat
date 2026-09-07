import { useState, useEffect } from "react";
import {
  Plus, Trash2, PanelLeftClose, Pin, Search, X,
  Folder, FolderPlus, ChevronRight, ChevronDown, Check,
} from "lucide-react";

// Mirrors the Rust `history::ConversationMeta` (snake_case over the wire).
export interface ConversationMeta {
  id: string;
  title: string;
  profile_id: string | null;
  model: string;
  created_at: number; // unix seconds
  updated_at: number;
  message_count: number;
  /// Pinned chats are returned first by the backend and shown in their own section at the top.
  pinned?: boolean;
  /// Folder the chat is filed under; null/absent = ungrouped.
  folder?: string | null;
  /// Display-only: a snippet of the search match, attached by App while a search is active.
  snippet?: string;
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
  onSetFolder: (id: string, folder: string | null) => void;
  searchQuery: string;
  onSearchChange: (q: string) => void;
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

const COLLAPSE_KEY = "lexi_collapsed_folders";

export function HistoryPanel({
  visible, conversations, activeId,
  onSelect, onNew, onDelete, onRename, onPin, onSetFolder,
  searchQuery, onSearchChange, onHide,
}: Props) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  // Which row's folder menu is open, and the in-progress "New folder…" input (row id + name).
  const [folderMenuId, setFolderMenuId] = useState<string | null>(null);
  const [newFolderFor, setNewFolderFor] = useState<string | null>(null);
  const [newFolderName, setNewFolderName] = useState("");
  // Collapsed folders persist across sessions — a folder you keep shut stays shut.
  const [collapsed, setCollapsed] = useState<Set<string>>(() => {
    try { return new Set(JSON.parse(localStorage.getItem(COLLAPSE_KEY) || "[]")); }
    catch { return new Set(); }
  });

  // Close an open folder menu on any outside click.
  useEffect(() => {
    if (!folderMenuId) return;
    const close = () => { setFolderMenuId(null); setNewFolderFor(null); };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [folderMenuId]);

  if (!visible) return null;

  const startRename = (c: ConversationMeta) => { setEditingId(c.id); setDraft(c.title); };
  const commitRename = () => {
    if (editingId) { const t = draft.trim(); if (t) onRename(editingId, t); }
    setEditingId(null);
  };

  const toggleCollapse = (f: string) => setCollapsed(prev => {
    const next = new Set(prev);
    next.has(f) ? next.delete(f) : next.add(f);
    try { localStorage.setItem(COLLAPSE_KEY, JSON.stringify([...next])); } catch { /* private mode */ }
    return next;
  });

  const assign = (id: string, folder: string | null) => {
    onSetFolder(id, folder);
    setFolderMenuId(null);
    setNewFolderFor(null);
    setNewFolderName("");
  };

  // All folder names in play, for the move-to menu.
  const allFolders = [...new Set(
    conversations.map(c => c.folder).filter((f): f is string => !!f),
  )].sort((a, b) => a.localeCompare(b));

  const renderItem = (c: ConversationMeta) => (
    <div
      key={c.id}
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
        {c.snippet && <div className="history-snippet">{c.snippet}</div>}
      </div>

      <button
        className={`history-folderbtn ${c.folder ? "on" : ""}`}
        title={c.folder ? `In folder "${c.folder}" — move…` : "Move to folder"}
        onClick={e => { e.stopPropagation(); setFolderMenuId(folderMenuId === c.id ? null : c.id); setNewFolderFor(null); }}
      >
        <Folder size={13} />
      </button>
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

      {folderMenuId === c.id && (
        <div className="history-folder-menu" onClick={e => e.stopPropagation()} onMouseDown={e => e.stopPropagation()}>
          {allFolders.map(f => (
            <button key={f} className="hfm-item" onClick={() => assign(c.id, f)}>
              <Folder size={12} />
              <span className="hfm-name">{f}</span>
              {c.folder === f && <Check size={12} className="hfm-check" />}
            </button>
          ))}
          {c.folder && (
            <button className="hfm-item hfm-remove" onClick={() => assign(c.id, null)}>
              <X size={12} /> Remove from folder
            </button>
          )}
          {newFolderFor === c.id ? (
            <input
              className="hfm-input"
              autoFocus
              placeholder="Folder name…"
              value={newFolderName}
              onChange={e => setNewFolderName(e.target.value)}
              onKeyDown={e => {
                if (e.key === "Enter") { const n = newFolderName.trim(); if (n) assign(c.id, n); }
                if (e.key === "Escape") { setNewFolderFor(null); setNewFolderName(""); }
              }}
            />
          ) : (
            <button className="hfm-item" onClick={() => { setNewFolderFor(c.id); setNewFolderName(""); }}>
              <FolderPlus size={12} /> New folder…
            </button>
          )}
        </div>
      )}
    </div>
  );

  // Grouping only applies to the full list. During a search the list is already filtered, so
  // show flat results (with their snippets) rather than scattering matches across sections.
  const searching = !!searchQuery.trim();
  const pinned = conversations.filter(c => c.pinned);
  const rest = conversations.filter(c => !c.pinned);
  const folderNames = [...new Set(
    rest.map(c => c.folder).filter((f): f is string => !!f),
  )].sort((a, b) => a.localeCompare(b));
  const ungrouped = rest.filter(c => !c.folder);

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

      <div className="history-search">
        <Search size={13} className="history-search-icon" />
        <input
          className="history-search-input"
          placeholder="Search chats…"
          value={searchQuery}
          onChange={e => onSearchChange(e.target.value)}
        />
        {searchQuery && (
          <button className="history-search-clear" title="Clear search" onClick={() => onSearchChange("")}>
            <X size={13} />
          </button>
        )}
      </div>

      <div className="history-list">
        {conversations.length === 0 && (
          <div className="history-empty">
            {searching ? "No chats match your search." : "No saved conversations yet."}
          </div>
        )}

        {searching ? (
          conversations.map(renderItem)
        ) : (
          <>
            {pinned.length > 0 && (
              <>
                <div className="history-section-label"><Pin size={11} /> Pinned</div>
                {pinned.map(renderItem)}
              </>
            )}

            {folderNames.map(f => {
              const items = rest.filter(c => c.folder === f);
              const isCollapsed = collapsed.has(f);
              return (
                <div key={f} className="history-folder-group">
                  <button className="history-folder-header" onClick={() => toggleCollapse(f)}>
                    {isCollapsed ? <ChevronRight size={13} /> : <ChevronDown size={13} />}
                    <Folder size={13} />
                    <span className="hf-name">{f}</span>
                    <span className="hf-count">{items.length}</span>
                  </button>
                  {!isCollapsed && items.map(renderItem)}
                </div>
              );
            })}

            {ungrouped.length > 0 && folderNames.length > 0 && (
              <div className="history-section-label">Ungrouped</div>
            )}
            {ungrouped.map(renderItem)}
          </>
        )}
      </div>
    </div>
  );
}

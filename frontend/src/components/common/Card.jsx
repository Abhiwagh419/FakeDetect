export default function Card({ title, subtitle, icon: Icon, action, children, className = "", style }) {
  return (
    <div className={`glass-panel p-5 ${className}`} style={style}>
      {(title || action) && (
        <div className="flex items-start justify-between mb-4 gap-3">
          <div className="flex items-center gap-2.5">
            {Icon && (
              <div className="w-8 h-8 rounded-lg bg-[var(--accent-dim)] flex items-center justify-center shrink-0">
                <Icon size={16} className="text-[var(--accent-strong)]" />
              </div>
            )}
            <div>
              {title && <h3 className="font-display font-semibold text-[15px] text-[var(--text)]">{title}</h3>}
              {subtitle && <p className="text-xs text-[var(--text-muted)] mt-0.5">{subtitle}</p>}
            </div>
          </div>
          {action}
        </div>
      )}
      {children}
    </div>
  );
}

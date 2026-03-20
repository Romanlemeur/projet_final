create table public.tracks (
  id              uuid primary key default gen_random_uuid(),
  user_id         uuid not null default auth.uid() references auth.users(id) on delete cascade,
  title           text not null,
  artist          text not null default '',
  filename        text not null,
  original_name   text not null,
  duration_seconds float not null default 0,
  bpm             float,
  key             text,
  mode            text,
  color           text not null default '#1DB954',
  created_at      timestamptz not null default now()
);

create table public.queue_items (
  id                  uuid primary key default gen_random_uuid(),
  user_id             uuid not null default auth.uid() references auth.users(id) on delete cascade,
  from_track_id       uuid not null references public.tracks(id) on delete cascade,
  to_track_id         uuid not null references public.tracks(id) on delete cascade,
  status              text not null default 'pending',
  transition_duration float not null default 20,
  transition_start    float,
  output_filename     text,
  created_at          timestamptz not null default now()
);

create table public.playlists (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid not null default auth.uid() references auth.users(id) on delete cascade,
  name       text not null,
  color      text not null default '#1DB954',
  created_at timestamptz not null default now()
);

create table public.playlist_items (
  id            uuid primary key default gen_random_uuid(),
  playlist_id   uuid not null references public.playlists(id) on delete cascade,
  queue_item_id uuid not null references public.queue_items(id) on delete cascade,
  position      int not null default 0,
  unique (playlist_id, queue_item_id)
);

alter table public.tracks        enable row level security;
alter table public.queue_items   enable row level security;
alter table public.playlists     enable row level security;
alter table public.playlist_items enable row level security;

create policy "tracks_select" on public.tracks for select using (auth.uid() = user_id);
create policy "tracks_insert" on public.tracks for insert with check (auth.uid() = user_id);
create policy "tracks_update" on public.tracks for update using (auth.uid() = user_id);
create policy "tracks_delete" on public.tracks for delete using (auth.uid() = user_id);

create policy "queue_select" on public.queue_items for select using (auth.uid() = user_id);
create policy "queue_insert" on public.queue_items for insert with check (auth.uid() = user_id);
create policy "queue_update" on public.queue_items for update using (auth.uid() = user_id);
create policy "queue_delete" on public.queue_items for delete using (auth.uid() = user_id);

create policy "playlists_select" on public.playlists for select using (auth.uid() = user_id);
create policy "playlists_insert" on public.playlists for insert with check (auth.uid() = user_id);
create policy "playlists_update" on public.playlists for update using (auth.uid() = user_id);
create policy "playlists_delete" on public.playlists for delete using (auth.uid() = user_id);

create policy "pitems_select" on public.playlist_items for select using (
  exists (select 1 from public.playlists where id = playlist_id and user_id = auth.uid())
);
create policy "pitems_insert" on public.playlist_items for insert with check (
  exists (select 1 from public.playlists where id = playlist_id and user_id = auth.uid())
);
create policy "pitems_delete" on public.playlist_items for delete using (
  exists (select 1 from public.playlists where id = playlist_id and user_id = auth.uid())
);
